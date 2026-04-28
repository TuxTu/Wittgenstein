"""
Core execution engine for model inference with activation patching and extraction.

This module provides the Executor class which handles:
- Running model forward/generate passes
- Applying activation patches recorded in a Prompt's modification ledger
- Extracting activation values from specific locations
- Resolving causal dependencies between ActivationRefs via their frozen
  dependency snapshots

The dependency model (replacing the old StateNode patch-lineage approach):
- Prompt writes are appended to ``Prompt._ledger``.  The executor reads the
  *effective* writes (latest per position) when running a forward pass.
- Every ActivationRef carries a ``dep_snapshot`` captured when an activation
  address is explicitly snapshotted.
- To evaluate an ActivationRef the executor re-runs its source prompt with
  exactly those snapshot writes applied — later ledger mutations do not
  retroactively change the ref's value.
"""
import threading
from dataclasses import dataclass

import torch
from typing import List, Optional, Callable, Dict, Tuple
from collections import defaultdict

from .computational_node import (
    ComputationalNode,
    ActivationAddress,
    ActivationAddressGroup,
    ActivationRef,
    ActivationRefGroup,
    WriteRecord,
)

# ---------------------------------------------------------------------------
# Thread-local active executor (like torch.no_grad())
# ---------------------------------------------------------------------------

_active_executor = threading.local()


def get_active_executor() -> Optional['Executor']:
    """Return the currently active executor, or ``None`` if outside a ``with executor:`` block."""
    return getattr(_active_executor, 'instance', None)


@dataclass
class PreparedWrite:
    """Concrete per-layer write ready to apply inside a hook."""

    token_indices: Tuple[int, ...]
    value: torch.Tensor


class Executor:
    """
    Core execution engine for running model inference with activation interventions.

    Args:
        model:      The HuggingFace transformer model.
        tokenizer:  The corresponding tokenizer.
        prompts:    A PromptList containing all prompts.
    """

    def __init__(self, model, tokenizer, prompts):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self._prev_executor = None
        self._prev_hidden_dim = None
        self._prev_num_layers = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._prev_executor = get_active_executor()
        self._prev_hidden_dim = ComputationalNode._HIDDEN_DIM_PLACEHOLDER
        self._prev_num_layers = ComputationalNode._NUM_LAYERS
        _active_executor.instance = self
        # Inject real model metadata into ComputationalNode class vars
        config = getattr(self.model, 'config', None)
        if config is not None:
            hidden_size = getattr(config, 'hidden_size', None)
            if hidden_size is not None:
                ComputationalNode._HIDDEN_DIM_PLACEHOLDER = hidden_size
        ComputationalNode._NUM_LAYERS = self.num_layers
        return self

    def __exit__(self, *exc):
        _active_executor.instance = self._prev_executor
        ComputationalNode._HIDDEN_DIM_PLACEHOLDER = self._prev_hidden_dim
        ComputationalNode._NUM_LAYERS = self._prev_num_layers
        return False

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers in the model."""
        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        if not base_model:
            base_model = self.model
        layers = getattr(base_model, "layers", None) or getattr(base_model, "h", None)
        if not layers:
            raise ValueError(f"Could not locate layers in model {type(self.model)}")
        return len(layers)

    def _input_device(self) -> torch.device:
        """Best-effort device for model inputs on single-device setups."""
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            pass

        device = getattr(self.model, "device", "cpu")
        return torch.device(device)

    def get_transformer_module(self, layer_idx: int, module_name: str):
        """
        Find the PyTorch module for a hook at the given layer and location.

        Supports common transformer architectures (LLaMA, GPT, etc.)
        """
        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        if not base_model:
            base_model = self.model

        layers = getattr(base_model, "layers", None) or getattr(base_model, "h", None)
        if not layers:
            raise ValueError(f"Could not locate layers in model {type(self.model)}")

        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of bounds")

        layer = layers[layer_idx]

        if module_name in ("resid_pre", "resid_post"):
            return layer
        elif module_name == "mlp":
            return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        elif module_name == "attn":
            return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)

        return None

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    @staticmethod
    def collect_leaves(node: ComputationalNode) -> List[ActivationRef]:
        """
        Recursively collect all unfilled ActivationRef leaves from a
        computational graph.  Delegates to :meth:`ComputationalNode.leaf_refs`.
        """
        return node.leaf_refs()

    @staticmethod
    def _normalize_extraction_targets(extraction_targets) -> Optional[List[ActivationRef]]:
        """
        Convert extraction target inputs into a flat list of ActivationRefs.

        Callers may provide a single ActivationAddress / ActivationRef, a
        group object, or an iterable of mixed address/ref targets.
        """
        if extraction_targets is None:
            return None

        def _flatten_one(target) -> List[ActivationRef]:
            if isinstance(target, ActivationAddress):
                return [target.snapshot()]
            if isinstance(target, ActivationAddressGroup):
                return list(target.snapshot()._refs)
            if isinstance(target, ActivationRef):
                return [target]
            if isinstance(target, ActivationRefGroup):
                return list(target._refs)
            raise TypeError(
                "Extraction targets must be activation addresses, activation "
                "snapshots, or iterables of those objects."
            )

        if isinstance(
            extraction_targets,
            (ActivationAddress, ActivationAddressGroup, ActivationRef, ActivationRefGroup),
        ):
            return _flatten_one(extraction_targets)

        normalized: List[ActivationRef] = []
        for target in extraction_targets:
            normalized.extend(_flatten_one(target))
        return normalized

    # ------------------------------------------------------------------
    # Dependency resolution via frozen snapshots
    # ------------------------------------------------------------------

    def fill_refs(self, refs: List[ActivationRef]) -> None:
        """
        Fill a list of ActivationRefs by running their source prompts.

        Refs are grouped by ``(prompt_id, snapshot_key)`` so that all refs
        sharing the same source prompt and the same causal snapshot are
        extracted in a single forward pass.

        For each group the snapshot's write value-nodes are recursively
        resolved before the forward pass runs.
        """
        # Group unfilled refs by (prompt_id, write-id tuple of their snapshot)
        groups: Dict[Tuple, List[ActivationRef]] = defaultdict(list)
        for ref in refs:
            if ref.evaluate() is not None:
                continue
            snapshot_key = tuple(wr.write_id for wr in ref.dep_snapshot)
            groups[(ref.prompt_id, snapshot_key)].append(ref)

        for (p_id, _), group_refs in groups.items():
            if p_id not in self.prompts:
                raise ValueError(
                    f"ActivationRef points to non-existent Prompt {p_id}"
                )
            dep_prompt = self.prompts.by_uid(p_id)
            snapshot_writes = group_refs[0].dep_snapshot  # Same for all in group

            # Recursively fill ActivationRefs nested inside the snapshot writes
            nested_unfilled: List[ActivationRef] = []
            for write in snapshot_writes:
                nested_unfilled.extend(self.collect_leaves(write.value_node))
            if nested_unfilled:
                self.fill_refs(nested_unfilled)

            print(
                f"[Dependency] Extracting {len(group_refs)} value(s) from P{p_id} "
                f"(snapshot writes: {len(snapshot_writes)})"
            )
            self.execute_pass(
                dep_prompt,
                mode="extraction",
                extraction_targets=group_refs,
                writes_to_apply=snapshot_writes,
            )

    # ------------------------------------------------------------------
    # Hook setup (shared by execute_pass and step)
    # ------------------------------------------------------------------

    def _setup_hooks(
        self,
        active_writes: List[WriteRecord],
        prompt_len: int,
        extraction_targets: Optional[List[ActivationRef]] = None,
    ) -> List:
        """
        Pre-resolve writes and extraction targets, create hook factories,
        and register hooks on model modules.

        Returns a list of hook handles that must be removed after the pass.
        """
        n_layers = self.num_layers

        # Group patches by hook location
        hook_to_writes: Dict[Tuple[int, str], List[PreparedWrite]] = defaultdict(list)
        for write in active_writes:
            prepared = self._prepare_write(write, prompt_len, n_layers)
            for li, prepared_write in prepared.items():
                hook_to_writes[(li, write.module)].append(prepared_write)

        # Group extraction targets by hook location
        hook_to_refs: Dict[Tuple, list] = defaultdict(list)
        if extraction_targets:
            for ref in extraction_targets:
                li = ref.layer_idx
                if li < 0:
                    li += n_layers
                hook_to_refs[(li, ref.module)].append((ref.token_idx, ref))

        # Hook factories
        def _make_pre_hook(layer_idx: int, module_name: str) -> Callable:
            local_writes = hook_to_writes.get((layer_idx, module_name), [])
            local_refs = hook_to_refs.get((layer_idx, module_name), [])

            def hook(module, args):
                tensor = args[0]
                if tensor.shape[1] != prompt_len:
                    return
                modified = False
                for prepared in local_writes:
                    if not modified:
                        tensor = tensor.clone()
                        modified = True
                    value = prepared.value.to(tensor.device, tensor.dtype)
                    if len(prepared.token_indices) == 1:
                        tensor[:, prepared.token_indices[0], :] = value
                    else:
                        tensor[:, list(prepared.token_indices), :] = value
                for tok_idx, ref in local_refs:
                    ref.set_cache(tensor[0, tok_idx, :].clone().detach())
                if modified:
                    return (tensor,) + args[1:]

            return hook

        def _make_post_hook(layer_idx: int, module_name: str) -> Callable:
            local_writes = hook_to_writes.get((layer_idx, module_name), [])
            local_refs = hook_to_refs.get((layer_idx, module_name), [])

            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.shape[1] != prompt_len:
                    return
                modified = False
                for prepared in local_writes:
                    if not modified:
                        tensor = tensor.clone()
                        modified = True
                    value = prepared.value.to(tensor.device, tensor.dtype)
                    if len(prepared.token_indices) == 1:
                        tensor[:, prepared.token_indices[0], :] = value
                    else:
                        tensor[:, list(prepared.token_indices), :] = value
                for tok_idx, ref in local_refs:
                    ref.set_cache(tensor[0, tok_idx, :].clone().detach())
                if modified:
                    if isinstance(output, tuple):
                        return (tensor,) + output[1:]
                    return tensor

            return hook

        # Register hooks
        needed_hooks: set = set(hook_to_writes.keys()) | set(hook_to_refs.keys())
        hook_handles = []
        for layer_idx, module_name in needed_hooks:
            mod = self.get_transformer_module(layer_idx, module_name)
            if mod is None:
                raise ValueError(
                    f"Model does not expose hook target {module_name!r} "
                    f"at layer {layer_idx}."
                )
            is_pre = (module_name == "resid_pre")
            if is_pre:
                h = mod.register_forward_pre_hook(
                    _make_pre_hook(layer_idx, module_name)
                )
            else:
                h = mod.register_forward_hook(
                    _make_post_hook(layer_idx, module_name)
                )
            hook_handles.append(h)

        return hook_handles

    def _expected_write_shape(
        self,
        token_count: int,
        layer_count: int,
        hidden_dim: int,
    ) -> List[int]:
        if layer_count > 1 and token_count > 1:
            return [layer_count, token_count, hidden_dim]
        if layer_count > 1:
            return [layer_count, hidden_dim]
        if token_count > 1:
            return [token_count, hidden_dim]
        return [hidden_dim]

    def _prepare_write(
        self,
        write: WriteRecord,
        prompt_len: int,
        n_layers: int,
    ) -> Dict[int, PreparedWrite]:
        """
        Resolve one logical write into concrete per-layer tensors.

        Supported RHS shapes are:
        - ``[D]`` broadcast to all selected positions
        - the exact logical target shape for the selected token/layer range
        """
        layer_indices = write.layer_selector.indices(n_layers)
        token_indices = tuple(write.token_selector.indices(prompt_len))
        value = write.value_node.evaluate()

        if value is None:
            raise RuntimeError(
                f"Write {write!r} could not be evaluated before hook setup."
            )
        if not torch.is_tensor(value):
            raise TypeError(
                f"Write {write!r} evaluated to {type(value).__name__}, expected torch.Tensor."
            )

        hidden_dim = ComputationalNode._HIDDEN_DIM_PLACEHOLDER
        token_count = len(token_indices)
        layer_count = len(layer_indices)

        if value.ndim == 0:
            raise ValueError(
                f"Write {write!r} produced scalar shape []; expected broadcast [D] "
                f"or exact logical shape {self._expected_write_shape(token_count, layer_count, hidden_dim)}."
            )
        if value.shape[-1] != hidden_dim:
            raise ValueError(
                f"Write {write!r} has hidden dim {value.shape[-1]}, expected {hidden_dim}."
            )

        prepared: Dict[int, PreparedWrite] = {}

        def _broadcast_vector(vec: torch.Tensor) -> Dict[int, PreparedWrite]:
            per_layer_value = (
                vec
                if token_count == 1
                else vec.unsqueeze(0).expand(token_count, -1).clone()
            )
            return {
                li: PreparedWrite(token_indices=token_indices, value=per_layer_value)
                for li in layer_indices
            }

        if value.ndim == 1:
            return _broadcast_vector(value)

        if layer_count == 1 and tuple(value.shape) == (token_count, hidden_dim):
            prepared[layer_indices[0]] = PreparedWrite(
                token_indices=token_indices,
                value=value if token_count > 1 else value[0],
            )
            return prepared

        if token_count == 1 and tuple(value.shape) == (layer_count, hidden_dim):
            for idx, li in enumerate(layer_indices):
                prepared[li] = PreparedWrite(
                    token_indices=token_indices,
                    value=value[idx],
                )
            return prepared

        if (
            layer_count > 1
            and token_count > 1
            and tuple(value.shape) == (layer_count, token_count, hidden_dim)
        ):
            for idx, li in enumerate(layer_indices):
                prepared[li] = PreparedWrite(
                    token_indices=token_indices,
                    value=value[idx],
                )
            return prepared

        raise ValueError(
            f"Write {write!r} has shape {list(value.shape)}; expected broadcast "
            f"[D] or exact logical shape "
            f"{self._expected_write_shape(token_count, layer_count, hidden_dim)}."
        )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute_pass(
        self,
        prompt,
        mode: str = "inference",
        max_new_tokens: Optional[int] = None,
        extraction_targets: Optional[List[ActivationRef]] = None,
        writes_to_apply: Optional[List[WriteRecord]] = None,
    ) -> Optional[str]:
        """
        Unified runner for model inference with patching and/or extraction.

        Args:
            prompt:
                The Prompt (or Chat) to run.
            mode:
                ``"inference"`` generates output text; ``"extraction"``
                runs a forward pass only and fills ``extraction_targets``.
            max_new_tokens:
                Token budget for generation (inference mode only).
            extraction_targets:
                ActivationRefs pointing to ``prompt`` that should be filled
                during this pass.
            writes_to_apply:
                Override the active writes for this run.  When ``None`` the
                prompt's own :meth:`~Prompt.effective_writes` are used.
                Pass a frozen dep_snapshot here when evaluating a ref.

        Returns:
            Generated text if ``mode="inference"``, else ``None``.
        """
        # Determine which writes are active for this run
        active_writes: List[WriteRecord] = (
            writes_to_apply if writes_to_apply is not None
            else prompt.effective_writes()
        )

        # Fill any ActivationRefs used as values inside active writes
        needed_refs: List[ActivationRef] = []
        for write in active_writes:
            needed_refs.extend(self.collect_leaves(write.value_node))
        if needed_refs:
            self.fill_refs(needed_refs)

        if active_writes:
            print(
                f"[Execute] Running P{prompt.uid} with {len(active_writes)} patch(es)..."
            )
        if mode == "extraction" and extraction_targets:
            print(
                f"[Execute] Extracting {len(extraction_targets)} value(s) from P{prompt.uid}..."
            )

        # Prepare input tensors
        token_ids_list = prompt.token_ids
        input_ids = torch.tensor(token_ids_list, device=self._input_device()).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        hook_handles = self._setup_hooks(active_writes, prompt_len, extraction_targets)

        try:
            with torch.no_grad():
                if mode == "inference":
                    pad_token_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id
                    )
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                    )
                    print(
                        self.tokenizer.decode(
                            output_ids[0][prompt_len:], skip_special_tokens=False
                        )
                    )
                    return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                else:
                    self.model(input_ids, attention_mask=attention_mask)
                    return None
        finally:
            for h in hook_handles:
                h.remove()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def eval(self, node: ComputationalNode) -> torch.Tensor:
        """
        Evaluate any computational node to a concrete tensor.

        Recursively fills all unfilled :class:`ActivationRef` leaves by running
        the required forward passes, then replays the recorded operations.

        Works for any node — an ``ActivationRef`` from a prompt, a composed
        expression like ``act_1 + act_2``, or a ``ConstantNode``.
        """
        leaves = node.leaf_refs()
        if leaves:
            self.fill_refs(leaves)
        result = node.evaluate()
        if result is None:
            raise RuntimeError(
                f"Node {node!r} evaluated to None after filling all leaves. "
                "This may indicate a node whose source prompt has no tokens, "
                "or an ActivationRef that was not reachable during the forward pass."
            )
        return result

    def generate(self, prompt, max_new_tokens: int = 35536) -> str:
        """
        Generate text for a prompt, applying any patches in its ledger.

        Args:
            prompt: A Prompt (or Chat) object.

        Returns:
            The generated text.
        """
        return self.execute_pass(prompt, mode="inference", max_new_tokens=max_new_tokens)

    def forward(self, prompt, extraction_targets=None) -> None:
        """
        Single forward pass — optionally fills extraction targets without
        generating text.

        If no targets are specified, this still runs a forward pass so that
        prompt patches take effect, but no frozen snapshots are extracted.
        """
        targets = self._normalize_extraction_targets(extraction_targets)
        self.execute_pass(
            prompt,
            mode="extraction",
            extraction_targets=targets,
        )

    def step(self, prompt) -> Tuple[int, str]:
        """
        Generate one token by running a single forward pass and sampling.

        Applies any patches from the prompt's ledger.  The predicted token
        is appended to ``prompt.tokens``.

        Returns:
            ``(token_id, token_str)`` of the generated token.
        """
        active_writes = prompt.effective_writes()

        # Fill any ActivationRefs used as values inside active writes
        needed_refs: List[ActivationRef] = []
        for write in active_writes:
            needed_refs.extend(self.collect_leaves(write.value_node))
        if needed_refs:
            self.fill_refs(needed_refs)

        token_ids_list = prompt.token_ids
        input_ids = torch.tensor(token_ids_list, device=self._input_device()).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        hook_handles = self._setup_hooks(active_writes, prompt_len)

        try:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(logits).item()
        finally:
            for h in hook_handles:
                h.remove()

        next_token_str = self.tokenizer.decode([next_token_id])
        prompt.append([(next_token_id, next_token_str)])
        return (next_token_id, next_token_str)
