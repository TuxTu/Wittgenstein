"""
Prompt storage classes for the witt library.
"""
from typing import Optional, List, Any, Tuple, Union, Dict, overload
import functools

from .computational_node import ComputationalNode, ActivationRef, ActivationRefGroup, ConstantNode, WriteRecord
from .selector import Selector, IndexSelector, SliceSelector, ListSelector


@functools.lru_cache(maxsize=1)
def _get_byte_decoder() -> Dict[str, int]:
    """
    Build the reverse mapping from GPT-2 BPE unicode characters to bytes.
    This is the inverse of the byte_encoder used in GPT-2/BPE tokenizers.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {chr(c): b for b, c in zip(bs, cs)}


def decode_bpe_token(token_str: str) -> str:
    """
    Decode a BPE token string to its actual text representation.
    Handles all GPT-2 style byte-level encodings (Ġ for space, Ċ for newline,
    em-dashes, curly quotes, etc.)
    """
    byte_decoder = _get_byte_decoder()
    try:
        byte_values = bytes([byte_decoder.get(c, ord(c)) for c in token_str])
        return byte_values.decode('utf-8', errors='replace')
    except Exception:
        return token_str


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
        ``prompt[tok][layer][module]`` instantiates an :class:`ActivationRef`
        that is cached in ``_node_registry`` — repeated or overlapping reads
        reuse the same object identity.
    """

    _next_uid: int = 0

    def __init__(self, tokens: Optional[List[Tuple[int, str]]] = None):
        self.tokens = tokens or []
        self.result: Any = None

        self.uid = Prompt._next_uid
        Prompt._next_uid += 1

        # Modification ledger: ordered list of all writes to this prompt.
        self._ledger: List[WriteRecord] = []
        self._write_id_counter: int = 0

        # Atomic node registry: (token_idx, layer_idx, module) → ActivationRef.
        # Overlapping selectors share the same atomic ref objects.
        self._node_registry: Dict[Tuple[int, int, str], ActivationRef] = {}

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

        Called by :meth:`LayerProxy.__setitem__`.  Later writes to the same
        ``(token_sel, layer_sel, module)`` key override earlier ones at
        inference time (see :meth:`effective_writes`).
        """
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

        affecting: List[WriteRecord] = []
        for wr in self.effective_writes():
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

    def get_or_instantiate_ref(
        self,
        token_sel: Selector,
        layer_sel: Selector,
        module: str,
    ) -> ComputationalNode:
        """
        Return the node for this coordinate, creating atomic refs on first
        access.

        Range selectors are decomposed into individual :class:`ActivationRef`
        objects (one per token/layer pair).  Overlapping ranges reuse the
        same atomic refs via ``_node_registry``.  Returns a single
        :class:`ActivationRef` for scalar selectors or an
        :class:`ActivationRefGroup` for ranges.

        Dependency snapshots are frozen per-atom at first instantiation.
        """

        n_tok = len(self.tokens)
        tok_indices = (
            [token_sel.resolve(n_tok)]
            if token_sel.is_single
            else token_sel.indices(n_tok)
        )

        layer_indices = layer_sel.indices_bounded()
        if layer_indices is None:
            if layer_sel.is_single:
                layer_indices = [layer_sel._index]
            else:
                raise ValueError(
                    f"Cannot decompose layer selector {layer_sel!r} without "
                    "knowing num_layers.  Use concrete non-negative bounds."
                )

        refs: list = []
        for li in layer_indices:
            for ti in tok_indices:
                atom_key = (ti, li, module)
                if atom_key not in self._node_registry:
                    dep = self.get_affecting_writes(
                        IndexSelector(ti), IndexSelector(li),
                    )
                    self._node_registry[atom_key] = ActivationRef(
                        prompt_id=self.uid,
                        token_idx=ti,
                        layer_idx=li,
                        module=module,
                        dep_snapshot=dep,
                    )
                refs.append(self._node_registry[atom_key])

        if len(refs) == 1:
            result: ComputationalNode = refs[0]
        else:
            result = ActivationRefGroup(
                refs,
                layer_count=len(layer_indices),
                token_count=len(tok_indices),
            )

        return result

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    def __getitem__(self, key: Union[int, slice, list]):
        """
        Access tokens by index, slice, or list of indices.

        - prompt[3]     -> TokenProxy with IndexSelector
        - prompt[3:7]   -> TokenProxy with SliceSelector
        - prompt[[0,2]] -> TokenProxy with ListSelector
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
        return TokenProxy(self, sel)

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
# Proxy chain:  Prompt -> TokenProxy -> LayerProxy -> ActivationRef
# ---------------------------------------------------------------------------

class TokenProxy:
    """
    Proxy for accessing token-level operations on a prompt.

    Stores a Selector for the token dimension.  The next ``__getitem__``
    call selects layers and produces a ``LayerProxy``.
    """

    def __init__(self, prompt: Prompt, token_selector: Selector):
        self.prompt = prompt
        self.token_selector = token_selector

    def __repr__(self) -> str:
        if self.token_selector.is_single:
            idx = self.token_selector._index
            decoded = decode_bpe_token(self.prompt.tokens[idx][1])
            return f"Token({idx}, {decoded!r})"
        return f"Token({self.token_selector})"

    def __getitem__(self, key: Union[int, slice, list]):
        """
        Select layer(s).

        - prompt[3][5]     -> LayerProxy with IndexSelector
        - prompt[3][2:5]   -> LayerProxy with SliceSelector
        - prompt[3][[1,3]] -> LayerProxy with ListSelector
        """
        layer_sel = Selector.from_key(key)
        return LayerProxy(self.prompt, self.token_selector, layer_sel)


class LayerProxy:
    """
    Proxy for accessing layer-level operations on token(s).

    Stores Selectors for both token and layer dimensions.

    - ``__getitem__(module)``  returns the :class:`ActivationRef` for this
      coordinate from the prompt's node registry (identity-preserving).
    - ``__setitem__(module, value)``  records a write to the prompt's ledger.
    """

    def __init__(self, prompt: Prompt, token_selector: Selector, layer_selector: Selector):
        self.prompt = prompt
        self.token_selector = token_selector
        self.layer_selector = layer_selector

    def __repr__(self) -> str:
        return f"Layer({self.layer_selector}, Token({self.token_selector}))"

    def __getitem__(self, module: str) -> ComputationalNode:
        """
        Return (or instantiate) the node for this coordinate.

        For single token/layer this is an :class:`ActivationRef`; for ranges
        it is an :class:`ActivationRefGroup` of atomic refs.  Snapshots are
        frozen per-atom at first instantiation.
        """
        return self.prompt.get_or_instantiate_ref(
            self.token_selector,
            self.layer_selector,
            module,
        )

    def __setitem__(self, module: str, value_node):
        """
        Record an activation write to the prompt's modification ledger.

        Any :class:`ComputationalNode` is accepted as the value.
        Raw scalars/tensors are wrapped in :class:`ConstantNode` automatically.
        """
        if not isinstance(value_node, ComputationalNode):
            value_node = ConstantNode(value_node)

        self.prompt.record_write(
            token_sel=self.token_selector,
            layer_sel=self.layer_selector,
            module=module,
            value_node=value_node,
        )


# ---------------------------------------------------------------------------
# PromptList
# ---------------------------------------------------------------------------

class PromptList:
    """A collection of prompts with lookup by uid."""

    def __init__(self):
        self._prompts: Dict[int, Prompt] = {}

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

        self._prompts[prompt.uid] = prompt
        return prompt

    def __getitem__(self, uid: int) -> Prompt:
        return self._prompts[uid]

    def __contains__(self, uid: int) -> bool:
        return uid in self._prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def __iter__(self):
        return iter(self._prompts.values())

    def __repr__(self) -> str:
        return f"PromptList({len(self._prompts)} prompts)"

    @property
    def last(self) -> Optional[Prompt]:
        """Get the most recently added prompt."""
        if not self._prompts:
            return None
        return list(self._prompts.values())[-1]
