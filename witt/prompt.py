"""
Prompt storage classes for the witt library.
"""
from typing import Optional, List, Any, Tuple, Union, Dict, overload

from .computational_node import (
    ComputationalNode,
    ActivationAddress,
    ActivationAddressGroup,
    ActivationRef,
    ActivationRefGroup,
    ConstantNode,
    WriteRecord,
)
from .selector import Selector, IndexSelector, SliceSelector, ListSelector
from .tokenize import decode_bpe_token


# ---------------------------------------------------------------------------
# Triangle-rule helpers for selector-based causal dependency
# ---------------------------------------------------------------------------

def _sel_min(sel: Selector, dim_size: Optional[int] = None) -> int:
    """
    Minimum index covered by a selector.

    Negative indices are resolved to their positive equivalent when
    ``dim_size`` is provided.  Returns a conservatively low value (0)
    when the minimum cannot be determined.
    """
    if isinstance(sel, IndexSelector):
        idx = sel._index
        if idx < 0 and dim_size is not None:
            idx += dim_size
        return idx
    elif isinstance(sel, SliceSelector):
        s = sel._slice
        step = s.step or 1
        if dim_size is not None:
            r = range(*s.indices(dim_size))
        elif s.start is not None and s.stop is not None:
            r = range(s.start, s.stop, step)
        else:
            return 0
        if len(r) == 0:
            return 0
        return r[0] if step > 0 else r[-1]
    elif isinstance(sel, ListSelector):
        indices = sel._indices
        if dim_size is not None:
            indices = [i + dim_size if i < 0 else i for i in indices]
        return min(indices)
    return 0


def _sel_max(sel: Selector, dim_size: Optional[int] = None) -> Optional[int]:
    """
    Maximum index covered by a selector.

    Returns ``None`` when the maximum cannot be determined (conservative:
    the caller treats ``None`` as "unbounded above").
    """
    if isinstance(sel, IndexSelector):
        idx = sel._index
        if idx < 0 and dim_size is not None:
            idx += dim_size
        return idx
    elif isinstance(sel, SliceSelector):
        s = sel._slice
        step = s.step or 1
        if dim_size is not None:
            r = range(*s.indices(dim_size))
        elif s.start is not None and s.stop is not None:
            r = range(s.start, s.stop, step)
        else:
            return None
        if len(r) == 0:
            return None
        return r[-1] if step > 0 else r[0]
    elif isinstance(sel, ListSelector):
        indices = sel._indices
        if dim_size is not None:
            indices = [i + dim_size if i < 0 else i for i in indices]
        return max(indices)
    return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

class Prompt:
    """
    Represents a stored input prompt with metadata.

    Attributes:
        tokens:  The tokenized sequence as (token_id, token_str) tuples.
        text:    (property) Raw input text reconstructed from tokens.
        uid:     Unique internal ID based on creation order (immutable).
        result:  Optional result from inspection.

    Ledger / registry:
        Every ``prompt[tok][layer][module] = value`` call appends a
        :class:`WriteRecord` to ``_ledger``.  Every read access
        ``prompt[tok][layer][module]`` returns a stable activation address
        cached in ``_address_registry``.  Call ``.snapshot()`` on that address
        to freeze the current causal write state into an :class:`ActivationRef`.
    """

    _next_uid: int = 0
    VALID_MODULE_NAMES = {"resid_pre", "resid_post", "mlp", "attn"}

    def __init__(self, tokens: Optional[List[Tuple[int, str]]] = None):
        self.tokens = tokens or []
        self.result: Any = None

        self.uid = Prompt._next_uid
        Prompt._next_uid += 1

        # Modification ledger: ordered list of all writes to this prompt.
        self._ledger: List[WriteRecord] = []
        self._write_id_counter: int = 0

        # Stable address registry: (token_idx, layer_idx, module) → ActivationAddress.
        # Overlapping selectors share the same atomic address objects.
        self._address_registry: Dict[Tuple[int, int, str], ActivationAddress] = {}

        # Group cache: (token_sel, layer_sel, module) → ActivationAddressGroup.
        # Ensures range accesses return the same group object on repeated calls.
        self._address_group_cache: Dict[Tuple, Union[ActivationAddress, ActivationAddressGroup]] = {}

    # ------------------------------------------------------------------
    # Token properties
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Reconstruct text from tokens by joining token strings."""
        return ''.join(decode_bpe_token(t[1]) for t in self.tokens)

    @property
    def token_ids(self) -> List[int]:
        """Return just the token IDs from the tokens list."""
        return [t[0] for t in self.tokens]

    @property
    def token_str(self) -> List[str]:
        """Return just the token strings from the tokens list."""
        return [decode_bpe_token(t[1]) for t in self.tokens]

    def __repr__(self) -> str:
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        new_line = '\n'
        tab = '\t'
        preview = preview.replace(new_line, '\\n').replace(tab, '\\t')
        return f"Prompt[{self.uid}]({preview!r})"

    def __str__(self) -> str:
        return self.text

    # ------------------------------------------------------------------
    # Ledger API
    # ------------------------------------------------------------------

    def record_write(
        self,
        token_sel: Selector,
        layer_sel: Selector,
        module: str,
        value_node: ComputationalNode,
    ) -> WriteRecord:
        """
        Append an activation write to the modification ledger.

        Called by :meth:`ActivationView.__setitem__`.  Later writes to the same
        ``(token_sel, layer_sel, module)`` key override earlier ones at
        inference time (see :meth:`effective_writes`).
        """
        self.validate_module_name(module)
        wr = WriteRecord(
            write_id=self._write_id_counter,
            token_selector=token_sel,
            layer_selector=layer_sel,
            module=module,
            value_node=value_node,
        )
        self._write_id_counter += 1
        self._ledger.append(wr)
        return wr

    def effective_writes(self) -> List[WriteRecord]:
        """
        Return the active set of writes for inference: latest write wins
        per ``(token_sel, layer_sel, module)`` key.

        Used by the Executor when running this prompt's forward pass.
        """
        seen: Dict[Tuple, WriteRecord] = {}
        for wr in self._ledger:
            key = (wr.token_selector, wr.layer_selector, wr.module)
            seen[key] = wr
        return list(seen.values())

    def get_affecting_writes(
        self,
        token_sel: Selector,
        layer_sel: Selector,
        effective_writes: Optional[List[WriteRecord]] = None,
    ) -> List[WriteRecord]:
        """
        Return the effective writes that causally affect ``(token_sel, layer_sel)``.

        Triangle rule (module-agnostic):
            A write at ``(write_tok, write_layer)`` affects target
            ``(target_tok, target_layer)`` iff::

                target_tok >= write_tok  AND  target_layer >= write_layer

        For selector ranges the check is conservative:
        a write is included if *any* write position could affect *any*
        target position, i.e.::

                max(target_range) >= min(write_range)

        The snapshot is taken over :meth:`effective_writes` (latest per key),
        so superseded writes are excluded.
        """
        n_tok = len(self.tokens)
        target_tok_max = _sel_max(token_sel, n_tok)
        target_layer_max = _sel_max(layer_sel, None)

        writes = effective_writes if effective_writes is not None else self.effective_writes()

        affecting: List[WriteRecord] = []
        for wr in writes:
            write_tok_min = _sel_min(wr.token_selector, n_tok)
            write_layer_min = _sel_min(wr.layer_selector, None)

            tok_ok = target_tok_max is None or target_tok_max >= write_tok_min
            layer_ok = target_layer_max is None or target_layer_max >= write_layer_min

            if tok_ok and layer_ok:
                affecting.append(wr)
        return affecting

    # ------------------------------------------------------------------
    # Node registry API
    # ------------------------------------------------------------------

    @classmethod
    def validate_module_name(cls, module: str) -> None:
        """Validate that the requested activation hook name is supported."""
        if module not in cls.VALID_MODULE_NAMES:
            allowed = ", ".join(sorted(cls.VALID_MODULE_NAMES))
            raise KeyError(
                f"Unknown activation module {module!r}. "
                f"Expected one of: {allowed}."
            )

    def _resolve_snapshot_layer_idx(self, layer_idx: int) -> int:
        """
        Resolve a layer index for snapshotting.

        Negative indices require the real layer count, which is injected while
        an executor is active.
        """
        n_layers = ComputationalNode._NUM_LAYERS
        resolved = layer_idx
        if resolved < 0:
            if n_layers is None:
                raise ValueError(
                    "Cannot snapshot a negative layer index without knowing "
                    "num_layers. Enter a 'with executor:' context or use a "
                    "non-negative layer index."
                )
            resolved += n_layers
        if n_layers is not None and (resolved < 0 or resolved >= n_layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range [-{n_layers}, {n_layers})"
            )
        return resolved

    def snapshot_address(self, address: ActivationAddress) -> ActivationRef:
        """Freeze an activation address against the prompt's current ledger."""
        return self.snapshot_addresses([address])[0]

    def snapshot_addresses(self, addresses: List[ActivationAddress]) -> List[ActivationRef]:
        """
        Freeze multiple activation addresses against one shared ledger snapshot.

        This avoids rescanning the full effective write set for every coordinate
        when a range of addresses is frozen together.
        """
        if not addresses:
            return []

        effective = self.effective_writes()
        refs: List[ActivationRef] = []
        for address in addresses:
            if address._prompt is not self:
                raise ValueError(
                    "All activation addresses in a batch snapshot must belong "
                    "to the same Prompt."
                )
            layer_idx = self._resolve_snapshot_layer_idx(address.layer_idx)
            dep = self.get_affecting_writes(
                IndexSelector(address.token_idx),
                IndexSelector(layer_idx),
                effective_writes=effective,
            )
            refs.append(
                ActivationRef(
                    prompt_id=self.uid,
                    token_idx=address.token_idx,
                    layer_idx=layer_idx,
                    module=address.module,
                    dep_snapshot=dep,
                )
            )
        return refs

    def snapshot_group(self, group: ActivationAddressGroup) -> ActivationRefGroup:
        """Freeze an activation address group into one ActivationRefGroup."""
        refs = self.snapshot_addresses(group._addresses)
        return ActivationRefGroup(
            refs,
            layer_count=group._layer_count,
            token_count=group._token_count,
        )

    def get_or_instantiate_address(
        self,
        token_sel: Selector,
        layer_sel: Selector,
        module: str,
    ) -> Union[ActivationAddress, ActivationAddressGroup]:
        """
        Return the stable address object for this coordinate, creating atomic
        addresses on first access.

        Range selectors are decomposed into individual
        :class:`ActivationAddress` objects (one per token/layer pair).
        Overlapping ranges reuse the same atomic addresses via
        ``_address_registry``.  Returns a single :class:`ActivationAddress`
        for scalar selectors or an :class:`ActivationAddressGroup` for ranges.
        """
        self.validate_module_name(module)

        # Check group cache for repeated range accesses
        group_key = (token_sel, layer_sel, module)
        if group_key in self._address_group_cache:
            return self._address_group_cache[group_key]

        n_tok = len(self.tokens)
        tok_indices = (
            [token_sel.resolve(n_tok)]
            if token_sel.is_single
            else token_sel.indices(n_tok)
        )

        layer_indices = layer_sel.indices_bounded()
        if layer_indices is None:
            n_layers = ComputationalNode._NUM_LAYERS
            if layer_sel.is_single:
                layer_indices = [layer_sel._index]
            elif n_layers is not None:
                layer_indices = layer_sel.indices(n_layers)
            else:
                raise ValueError(
                    f"Cannot decompose layer selector {layer_sel!r} without "
                    "knowing num_layers.  Use concrete non-negative bounds "
                    "or enter a 'with executor:' context."
                )

        addresses: list = []
        for li in layer_indices:
            for ti in tok_indices:
                atom_key = (ti, li, module)
                if atom_key not in self._address_registry:
                    self._address_registry[atom_key] = ActivationAddress(
                        prompt=self,
                        token_idx=ti,
                        layer_idx=li,
                        module=module,
                    )
                addresses.append(self._address_registry[atom_key])

        if len(addresses) == 1:
            result: Union[ActivationAddress, ActivationAddressGroup] = addresses[0]
        else:
            result = ActivationAddressGroup(
                addresses,
                layer_count=len(layer_indices),
                token_count=len(tok_indices),
            )

        self._address_group_cache[group_key] = result
        return result

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    def __getitem__(self, key: Union[int, slice, list]):
        """
        Access tokens by index, slice, or list of indices.

        - prompt[3]     -> ActivationView with IndexSelector
        - prompt[3:7]   -> ActivationView with SliceSelector
        - prompt[[0,2]] -> ActivationView with ListSelector
        """
        sel = Selector.from_key(key)
        n = len(self.tokens)
        if sel.is_single:
            idx = key if isinstance(key, int) else key
            if idx < -n or idx >= n:
                raise IndexError(
                    f"Token index {idx} out of range [-{n}, {n})"
                )
        if isinstance(key, slice):
            concrete = slice(*key.indices(n))
            sel = SliceSelector(concrete)
        return ActivationView(self, sel)

    def generate(self, **kwargs) -> str:
        """Generate text using the active executor. Requires ``with executor:`` context."""
        from .executor import get_active_executor
        exe = get_active_executor()
        if exe is None:
            raise RuntimeError(
                "No active executor. Use 'with executor:' to set one."
            )
        return exe.generate(self, **kwargs)

    def forward(self, extraction_targets=None):
        """
        Single forward pass via the active executor.

        Optionally accepts extraction targets as activation addresses,
        activation snapshots, or iterables of those objects. Requires
        ``with executor:`` context.
        """
        from .executor import get_active_executor
        exe = get_active_executor()
        if exe is None:
            raise RuntimeError(
                "No active executor. Use 'with executor:' to set one."
            )
        return exe.forward(self, extraction_targets=extraction_targets)

    def step(self):
        """Generate one token via the active executor. Requires ``with executor:`` context."""
        from .executor import get_active_executor
        exe = get_active_executor()
        if exe is None:
            raise RuntimeError(
                "No active executor. Use 'with executor:' to set one."
            )
        return exe.step(self)

    def append(self, new_tokens: List[Tuple[int, str]]):
        self.tokens.extend(new_tokens)

    def find_content_index(self, content: str) -> Tuple[int, int]:
        return self.extract_message(content, self.token_str)

    def extract_message(self, content: str, tokens: List[str]) -> Tuple[int, int]:
        """
        Find the start and end token indices that contain the given message content.

        Uses full BPE byte decoding to properly handle all special characters
        (em-dashes, curly quotes, etc.)
        """
        if not tokens or not content:
            return (-1, -1)

        char_to_token = []
        reconstructed = ""

        for token_idx, token_str in enumerate(tokens):
            decoded = decode_bpe_token(token_str)
            for _ in decoded:
                char_to_token.append(token_idx)
            reconstructed += decoded

        pos = reconstructed.find(content)
        if pos == -1:
            return (-1, -1)

        start_idx = char_to_token[pos]
        end_idx = char_to_token[pos + len(content) - 1]

        return (start_idx, end_idx)


# ---------------------------------------------------------------------------
# Proxy chain:  Prompt -> ActivationView -> ActivationView -> ActivationAddress
# ---------------------------------------------------------------------------

class ActivationView(ComputationalNode):
    """
    Progressive view into a prompt's activation space.

    This is a :class:`ComputationalNode` subclass so that token/layer
    selection participates naturally in the lazy graph API until the final
    module name is chosen.  At that point, indexing returns a stable
    activation address object which can later be frozen via ``.snapshot()``.

    Indexing chain::

        Prompt[tok]         → ActivationView(token_sel, layer_sel=None)
        view[layer]         → ActivationView(token_sel, layer_sel)
        view["module"]      → ActivationAddress or ActivationAddressGroup
        view["module"] = v  → WriteRecord appended to prompt ledger
    """

    def __init__(self, prompt: Prompt, token_selector: Selector, layer_selector: Optional[Selector] = None):
        self._prompt = prompt
        self._token_sel = token_selector
        self._layer_sel = layer_selector
        self._meta = None  # Partial view — no concrete tensor shape yet

    # -- Override OperatorMeta-injected equality to keep identity semantics --
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    def __repr__(self) -> str:
        if self._layer_sel is None:
            if self._token_sel.is_single:
                idx = self._token_sel._index
                decoded = decode_bpe_token(self._prompt.tokens[idx][1])
                return f"View(P{self._prompt.uid}, tok={idx}, {decoded!r})"
            return f"View(P{self._prompt.uid}, tok={self._token_sel})"
        return f"View(P{self._prompt.uid}, tok={self._token_sel}, layer={self._layer_sel})"

    # -- Progressive indexing (overrides ComputationalNode.__getitem__) --

    def __getitem__(self, key):
        """
        Progressive narrowing of the activation space.

        - If layer not yet selected: interpret *key* as a layer selector.
        - If layer is selected and *key* is a ``str``: resolve to
          :class:`ActivationAddress` / :class:`ActivationAddressGroup`.
        """
        if self._layer_sel is None:
            # Second bracket: select layer(s)
            layer_sel = Selector.from_key(key)
            # Validate against real layer count when available
            n_layers = ComputationalNode._NUM_LAYERS
            if n_layers is not None and layer_sel.is_single:
                idx = layer_sel._index
                if idx < -n_layers or idx >= n_layers:
                    raise IndexError(
                        f"Layer index {idx} out of range [-{n_layers}, {n_layers})"
                    )
            return ActivationView(self._prompt, self._token_sel, layer_sel)

        if isinstance(key, str):
            # Third bracket: module name → resolve to stable address object
            return self._prompt.get_or_instantiate_address(
                self._token_sel,
                self._layer_sel,
                key,
            )

        raise TypeError(
            f"Expected a module name (str), got {type(key).__name__}. "
            "Use view[layer]['module_name'] to access activations."
        )

    def __setitem__(self, key, value_node):
        """
        Record an activation write to the prompt's modification ledger.

        Requires that both token and layer selectors are set.
        Raw scalars/tensors are wrapped in :class:`ConstantNode` automatically.
        """
        if self._layer_sel is None:
            raise TypeError(
                "Layer not yet selected. "
                "Use prompt[token][layer]['module'] = value."
            )
        if not isinstance(key, str):
            raise TypeError(
                f"Module key must be a str, got {type(key).__name__}."
            )
        if isinstance(value_node, (ActivationAddress, ActivationAddressGroup)):
            raise TypeError(
                "Activation addresses are not frozen values. Call .snapshot() "
                "before assigning them."
            )
        if not isinstance(value_node, ComputationalNode):
            value_node = ConstantNode(value_node)

        self._prompt.record_write(
            token_sel=self._token_sel,
            layer_sel=self._layer_sel,
            module=key,
            value_node=value_node,
        )

    # -- Block stray attribute access (overrides ComputationalNode.__getattr__) --

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        raise AttributeError(
            f"ActivationView has no attribute '{name}'. "
            "Use [layer_index] then ['module_name'] to access activations."
        )

    # -- Block arithmetic on views (overrides ComputationalNode.__torch_function__) --

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        for a in args:
            if isinstance(a, ActivationView):
                raise TypeError(
                    "Cannot perform tensor operations on an ActivationView. "
                    "Fully resolve to an activation snapshot first: "
                    "view[layer]['module'].snapshot()"
                )
        return super().__torch_function__(func, types, args, kwargs)

    # -- A view is not a concrete value --

    def evaluate(self):
        return None

    def children(self):
        return []


# ---------------------------------------------------------------------------
# PromptList
# ---------------------------------------------------------------------------

class PromptList:
    """An ordered collection of prompts with explicit lookup by uid."""

    def __init__(self):
        self._prompts: Dict[int, Prompt] = {}
        self._order: List[int] = []

    @overload
    def add(self, prompt: Prompt) -> Prompt: ...

    @overload
    def add(self, tokens: List[Tuple[int, str]]) -> Prompt: ...

    def add(self, tokens_or_prompt: Union[List[Tuple[int, str]], Prompt]) -> Prompt:
        """
        Add a new prompt to the collection.

        Args:
            tokens_or_prompt: Either a list of (token_id, token_str) tuples
                              or an existing Prompt object.

        Returns:
            The added Prompt object.
        """
        if isinstance(tokens_or_prompt, Prompt):
            prompt = tokens_or_prompt
        else:
            prompt = Prompt(tokens=tokens_or_prompt)

        if prompt.uid not in self._prompts:
            self._order.append(prompt.uid)
        self._prompts[prompt.uid] = prompt
        return prompt

    def by_uid(self, uid: int) -> Prompt:
        """Look up a prompt by its immutable uid."""
        return self._prompts[uid]

    def __getitem__(self, key: Union[int, slice]) -> Union[Prompt, List[Prompt]]:
        """
        Positional access by insertion order.

        - ``prompts[0]`` returns the first prompt
        - ``prompts[-1]`` returns the most recent prompt
        - ``prompts[1:3]`` returns a list of prompts
        """
        if isinstance(key, slice):
            uids = self._order[key]
            return [self._prompts[uid] for uid in uids]
        if isinstance(key, int):
            uid = self._order[key]
            return self._prompts[uid]
        raise TypeError(f"PromptList indices must be int or slice, got {type(key).__name__}")

    def __contains__(self, item) -> bool:
        if isinstance(item, Prompt):
            return item.uid in self._prompts
        return item in self._prompts

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self):
        return (self._prompts[uid] for uid in self._order)

    def __repr__(self) -> str:
        return f"PromptList({len(self._prompts)} prompts)"

    @property
    def last(self) -> Optional[Prompt]:
        """Get the most recently added prompt."""
        if not self._order:
            return None
        return self._prompts[self._order[-1]]
