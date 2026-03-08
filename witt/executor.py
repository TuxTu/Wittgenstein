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
- Every ActivationRef carries a ``dep_snapshot``: the subset of writes that
  causally affected its position at the moment of first instantiation.
- To evaluate an ActivationRef the executor re-runs its source prompt with
  exactly those snapshot writes applied — later ledger mutations do not
  retroactively change the ref's value.
"""
import torch
from typing import List, Optional, Callable, Dict, Tuple
from collections import defaultdict

from .computational_node import ComputationalNode, ActivationRef, WriteRecord


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
        computational graph.

        Uses the generic ``children()`` interface so it works with all node
        types without type-specific checks.
        """
        if isinstance(node, ActivationRef) and node.evaluate() is None:
            return [node]
        leaves: List[ActivationRef] = []
        for child in node.children():
            leaves.extend(Executor.collect_leaves(child))
        return leaves

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
            dep_prompt = self.prompts[p_id]
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
        input_ids = torch.tensor(token_ids_list, device=self.model.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        n_layers = self.num_layers

        # Pre-resolve and group patches and targets by hook location
        # This avoids evaluating selectors and looping over unrelated writes inside the hot PyTorch hook
        hook_to_writes = defaultdict(list)
        for write in active_writes:
            write_layer_indices = write.layer_selector.indices(n_layers)
            tok_idx = write.token_selector.resolve(prompt_len)
            for li in write_layer_indices:
                hook_to_writes[(li, write.module)].append((tok_idx, write))

        # Atomic refs: resolve negative layer indices and group by hook site
        hook_to_refs = defaultdict(list)
        if mode == "extraction" and extraction_targets:
            for ref in extraction_targets:
                li = ref.layer_idx
                if li < 0:
                    li += n_layers
                hook_to_refs[(li, ref.module)].append((ref.token_idx, ref))

        # ------------------------------------------------------------------
        # Hook factories (separate signatures for pre- vs post-hooks)
        # ------------------------------------------------------------------

        def _make_pre_hook(layer_idx: int, module_name: str) -> Callable:
            local_writes = hook_to_writes.get((layer_idx, module_name), [])
            local_refs = hook_to_refs.get((layer_idx, module_name), [])

            def hook(module, args):
                tensor = args[0]
                if tensor.shape[1] != prompt_len:
                    return
                modified = False
                for tok_idx, write in local_writes:
                    val = write.value_node.evaluate()
                    if val is not None:
                        if not modified:
                            tensor = tensor.clone()
                            modified = True
                        tensor[:, tok_idx, :] = val.to(tensor.device, tensor.dtype)
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
                for tok_idx, write in local_writes:
                    val = write.value_node.evaluate()
                    if val is not None:
                        if not modified:
                            tensor = tensor.clone()
                            modified = True
                        tensor[:, tok_idx, :] = val.to(tensor.device, tensor.dtype)
                for tok_idx, ref in local_refs:
                    ref.set_cache(tensor[0, tok_idx, :].clone().detach())
                if modified:
                    if isinstance(output, tuple):
                        return (tensor,) + output[1:]
                    return tensor

            return hook

        # ------------------------------------------------------------------
        # Register hooks only at required (layer, module) pairs
        # ------------------------------------------------------------------

        needed_hooks: set = set(hook_to_writes.keys()) | set(hook_to_refs.keys())

        hook_handles = []
        for layer_idx, module_name in needed_hooks:
            mod = self.get_transformer_module(layer_idx, module_name)
            if mod is None:
                continue
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

        # ------------------------------------------------------------------
        # Run model
        # ------------------------------------------------------------------

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

    def generate(self, prompt, max_new_tokens: int = 35536) -> str:
        """
        Generate text for a prompt, applying any patches in its ledger.

        Args:
            prompt: A Prompt (or Chat) object.

        Returns:
            The generated text.
        """
        return self.execute_pass(prompt, mode="inference", max_new_tokens=max_new_tokens)
