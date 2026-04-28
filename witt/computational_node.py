"""
Lazy computational graph for activation operations.

ComputationalNode is a generic recording proxy: any operation performed on it
(arithmetic, torch functions, method calls, indexing) is recorded as a new graph
node and replayed at evaluation time. Operations are validated at definition time
via meta tensor dry-runs.
"""
import torch
import operator as pyop
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Union, Tuple, Optional

from .selector import Selector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_meta(obj):
    """Convert to meta tensor for shape validation."""
    if isinstance(obj, ComputationalNode):
        return obj._meta
    if isinstance(obj, torch.Tensor):
        return obj.to(device='meta')
    return obj




def _resolve_args(args: tuple) -> tuple:
    """Recursively evaluate any ComputationalNode found in args."""
    return tuple(a.evaluate() if isinstance(a, ComputationalNode) else a for a in args)


def _resolve_kwargs(kwargs: dict) -> dict:
    """Recursively evaluate any ComputationalNode found in kwargs."""
    return {k: v.evaluate() if isinstance(v, ComputationalNode) else v
            for k, v in kwargs.items()}


def _resolve_key(key):
    """Recursively evaluate any ComputationalNode found in an index key."""
    if isinstance(key, ComputationalNode):
        return key.evaluate()
    if isinstance(key, tuple):
        return tuple(_resolve_key(k) for k in key)
    return key


def _build_operator_mappings():
    """Automatically build operator mappings by scanning operator module."""
    # Auto-discover operator functions from operator module
    # Pattern: most map operator.func → __func__ dunder
    # Special cases handled in overrides
    dunder_to_op = {}
    op_to_torch = {}
    unary_ops = set()

    # Known unary operators in operator module
    UNARY_OPERATOR_NAMES = {'neg', 'pos', 'abs', 'invert'}

    # Discover all callable attributes in operator module
    for name in dir(pyop):
        if name.startswith('_'):  # Skip private attributes
            continue

        op_func = getattr(pyop, name)
        if not callable(op_func):
            continue

        # Convert operator function name to dunder name
        # Handle special cases
        if name == 'and_':
            dunder = '__and__'
            torch_name = 'bitwise_and'
        elif name == 'or_':
            dunder = '__or__'
            torch_name = 'bitwise_or'
        elif name == 'xor':
            dunder = '__xor__'
            torch_name = 'bitwise_xor'
        elif name == 'invert':
            dunder = '__invert__'
            torch_name = 'bitwise_not'
        elif name == 'lt':
            dunder = '__lt__'
            torch_name = 'less'
        elif name == 'le':
            dunder = '__le__'
            torch_name = 'less_equal'
        elif name == 'gt':
            dunder = '__gt__'
            torch_name = 'greater'
        elif name == 'ge':
            dunder = '__ge__'
            torch_name = 'greater_equal'
        elif name == 'truediv':
            dunder = '__truediv__'
            torch_name = 'true_divide'
        elif name == 'floordiv':
            dunder = '__floordiv__'
            torch_name = 'floor_divide'
        elif name == 'mod':
            dunder = '__mod__'
            torch_name = 'remainder'
        elif name == 'pos':
            dunder = '__pos__'
            torch_name = 'positive'  # may not exist
        else:
            # Default pattern: operator.add → __add__
            dunder = f'__{name}__'
            torch_name = name

        # Get torch function (handle missing functions gracefully)
        torch_func = getattr(torch, torch_name, None)
        if torch_func is None:
            if name == 'pos':
                torch_func = lambda x: x  # identity function
            else:
                continue  # Skip if torch doesn't have this operator

        dunder_to_op[dunder] = op_func
        op_to_torch[op_func] = torch_func

        # Track unary operators
        if name in UNARY_OPERATOR_NAMES:
            unary_ops.add(dunder)

    return dunder_to_op, op_to_torch, unary_ops


# ---------------------------------------------------------------------------
# Metaclass for operator support
# ---------------------------------------------------------------------------

class OperatorMeta(type):
    """
    Metaclass that automatically adds Python operator support to ComputationalNode.
    All operators delegate to torch functions, which then trigger __torch_function__
    to create TorchFunctionNode instances.
    """

    # Build mappings once using module-level function
    _DUNDER_TO_OP, _OP_TO_TORCH, _UNARY_OPS = _build_operator_mappings()

    @classmethod
    def _make_forward_method(cls, torch_func, is_unary=False):
        """Create a forward operator method that delegates to torch function."""
        if is_unary:
            def method(self):
                return torch_func(self)  # Triggers __torch_function__
        else:
            def method(self, other):
                return torch_func(self, other)  # Triggers __torch_function__
        return method

    @classmethod
    def _make_reverse_method(cls, torch_func):
        """Create a reverse operator method (e.g., __radd__)."""
        def method(self, other):
            # Swap arguments: torch_func(other, self)
            return torch_func(other, self)  # Triggers __torch_function__
        return method

    def __new__(cls, name, bases, namespace):
        # Add forward operators
        for dunder, pyop_func in cls._DUNDER_TO_OP.items():
            if pyop_func not in cls._OP_TO_TORCH:
                continue
            torch_func = cls._OP_TO_TORCH[pyop_func]
            is_unary = dunder in cls._UNARY_OPS
            namespace[dunder] = cls._make_forward_method(torch_func, is_unary)

        # Add reverse operators (for binary operators only)
        reverse_map = {}
        for dunder in cls._DUNDER_TO_OP:
            if dunder in cls._UNARY_OPS:
                continue  # Skip unary operators
            # Convert __add__ to __radd__, __sub__ to __rsub__, etc.
            # Pattern: __{op}__ -> __r{op}__
            op_name = dunder[2:-2]  # Remove '__' prefix and suffix
            rev_dunder = f'__r{op_name}__'
            reverse_map[rev_dunder] = dunder

        for rev_dunder, fwd_dunder in reverse_map.items():
            if fwd_dunder in cls._DUNDER_TO_OP:
                pyop_func = cls._DUNDER_TO_OP[fwd_dunder]
                if pyop_func in cls._OP_TO_TORCH:
                    torch_func = cls._OP_TO_TORCH[pyop_func]
                    namespace[rev_dunder] = cls._make_reverse_method(torch_func)

        return super().__new__(cls, name, bases, namespace)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ComputationalNode(metaclass=OperatorMeta):
    """
    Base class for every node in the lazy computation graph.

    Intercepts operations via three mechanisms:
      - __torch_function__  : catches torch.xxx(node, ...)
      - __getattr__         : catches node.method(...) and node.attr
      - __getitem__         : catches node[key]
    Arithmetic dunders are automatically generated via OperatorMeta metaclass.

    Every node carries a ``_meta`` tensor (device='meta', zero memory) that
    tracks the output shape.  When a new node is created the same operation
    runs on the meta tensors so shape errors surface immediately.
    """

    # Fixed placeholder for shape validation.  All activations share the same
    # hidden_dim so the actual value never causes incompatibilities.
    # Inside a ``with executor:`` block these are set to real model values.
    _HIDDEN_DIM_PLACEHOLDER: ClassVar[int] = 64
    _NUM_LAYERS: ClassVar[Optional[int]] = None

    _meta: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Generic graph traversal
    # ------------------------------------------------------------------

    def children(self) -> List['ComputationalNode']:
        """Return child nodes for graph traversal. Override in subclasses."""
        return []

    def leaf_refs(self) -> List['ActivationRef']:
        """Recursively collect all unfilled ActivationRef leaves from this node's graph."""
        if isinstance(self, ActivationRef) and self.evaluate() is None:
            return [self]
        leaves: List['ActivationRef'] = []
        for child in self.children():
            leaves.extend(child.leaf_refs())
        return leaves

    # ------------------------------------------------------------------
    # Evaluate (replays the recorded operation on real tensors)
    # ------------------------------------------------------------------

    def evaluate(self) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def eval(self) -> torch.Tensor:
        """
        Evaluate this node to a concrete tensor via the active executor.

        Fills all unfilled leaf dependencies first.  Requires ``with executor:``.
        """
        from .executor import get_active_executor
        exe = get_active_executor()
        if exe is None:
            raise RuntimeError(
                "No active executor. Use 'with executor:' to set one."
            )
        return exe.eval(self)

    # ------------------------------------------------------------------
    # __torch_function__  –  intercepts torch.xxx(node, ...)
    # ------------------------------------------------------------------

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        node = TorchFunctionNode(func, args, kwargs)
        # Validate via meta tensors.  Some ops don't support the Meta backend
        # in older PyTorch versions – fall back gracefully (skip shape tracking).
        try:
            meta_args = tuple(_to_meta(a) for a in args)
            meta_kwargs = {k: _to_meta(v) for k, v in kwargs.items()}
            node._meta = func(*meta_args, **meta_kwargs)
        except NotImplementedError:
            # Meta backend not available for this op – skip validation
            node._meta = None
        return node

    # ------------------------------------------------------------------
    # __getattr__  –  intercepts node.method(...) and node.attr
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # Private / dunder attributes must NOT be intercepted
        if name.startswith('_'):
            raise AttributeError(name)
        return AttrAccessNode(self, name)

    # ------------------------------------------------------------------
    # __getitem__  –  intercepts node[key]  (post-hoc tensor indexing)
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        node = IndexNode(self, key)
        if self._meta is not None:
            try:
                node._meta = self._meta[key]
            except (NotImplementedError, RuntimeError, IndexError) as e:
                if isinstance(e, IndexError):
                    raise  # Real index errors should propagate
                node._meta = None
        else:
            node._meta = None
        return node



# ---------------------------------------------------------------------------
# Write record (used by Prompt ledger and ActivationRef snapshots)
# ---------------------------------------------------------------------------

@dataclass
class WriteRecord:
    """
    A single activation write recorded in a Prompt's modification ledger.

    Captured once per ``prompt[token][layer][module] = value`` call.
    Frozen activation snapshots capture the subset of WriteRecords that
    causally affect their position at snapshot time.
    """
    write_id: int
    token_selector: 'Selector'
    layer_selector: 'Selector'
    module: str
    value_node: 'ComputationalNode'


# ---------------------------------------------------------------------------
# Stable activation addresses
# ---------------------------------------------------------------------------

class _AddressErrorMixin:
    """Shared guardrails for stable activation addresses."""

    __eq__ = object.__eq__
    __hash__ = object.__hash__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise TypeError(
            "Activation addresses are not frozen values. Call .snapshot() "
            "before using them in tensor operations."
        )

    def eval(self):
        raise TypeError(
            "Activation addresses cannot be evaluated directly. "
            "Call .snapshot().eval() instead."
        )


class ActivationAddress(_AddressErrorMixin):
    """
    Stable pointer to one activation coordinate.

    Unlike :class:`ActivationRef`, an address does not freeze causal writes.
    Call :meth:`snapshot` to create a fresh frozen ref from the prompt's
    current ledger state.
    """

    def __init__(self, prompt, token_idx: int, layer_idx: int, module: str):
        self._prompt = prompt
        self.prompt_id = prompt.uid
        self.token_idx = token_idx
        self.layer_idx = layer_idx
        self.module = module

    def snapshot(self) -> 'ActivationRef':
        """Freeze the address against the prompt's current effective writes."""
        return self._prompt.snapshot_address(self)

    def __repr__(self) -> str:
        return (
            f"Addr(P{self.prompt_id}"
            f".T{self.token_idx}.L{self.layer_idx}.{self.module})"
        )


class ActivationAddressGroup(_AddressErrorMixin):
    """
    Stable collection of activation addresses created from a token/layer range.

    Call :meth:`snapshot` to freeze the full group into an
    :class:`ActivationRefGroup`.
    """

    def __init__(
        self,
        addresses: List['ActivationAddress'],
        layer_count: int,
        token_count: int,
    ):
        self._addresses = addresses
        self._layer_count = layer_count
        self._token_count = token_count

    def snapshot(self) -> 'ActivationRefGroup':
        """Freeze each address in the group into a fresh ActivationRefGroup."""
        if not self._addresses:
            raise ValueError("Cannot snapshot an empty ActivationAddressGroup")
        return self._addresses[0]._prompt.snapshot_group(self)

    def __repr__(self) -> str:
        return (
            f"AddrGroup(layers={self._layer_count}, tokens={self._token_count}, "
            f"addrs={len(self._addresses)})"
        )


# ---------------------------------------------------------------------------
# Frozen snapshot nodes
# ---------------------------------------------------------------------------

class ActivationRef(ComputationalNode):
    """
    Frozen snapshot of one activation at a single (token, layer, module) position.

    This is a leaf node: it holds no computation, only an address plus the
    causal write snapshot captured when :meth:`ActivationAddress.snapshot`
    was called.  The Executor fills in the actual tensor value via
    :meth:`set_cache` during a forward pass.
    """

    def __init__(
        self,
        prompt_id: int,
        token_idx: int,
        layer_idx: int,
        module: str,
        dep_snapshot: List['WriteRecord'],
    ):
        self.prompt_id = prompt_id
        self.token_idx = token_idx
        self.layer_idx = layer_idx
        self.module = module
        self.dep_snapshot: List['WriteRecord'] = dep_snapshot
        self._runtime_cache: Optional[torch.Tensor] = None

        D = ComputationalNode._HIDDEN_DIM_PLACEHOLDER
        self._meta = torch.empty(D, device='meta')

    def evaluate(self) -> Optional[torch.Tensor]:
        return self._runtime_cache

    def set_cache(self, activation: torch.Tensor):
        """Called by the Executor inside a hook to fill this ref."""
        self._runtime_cache = activation

    def children(self) -> List[ComputationalNode]:
        return []

    def __repr__(self):
        return (
            f"Ref(P{self.prompt_id}"
            f".T{self.token_idx}.L{self.layer_idx}.{self.module}"
            f"[deps={len(self.dep_snapshot)}])"
        )


class ActivationRefGroup(ComputationalNode):
    """
    Composite node assembling multiple atomic :class:`ActivationRef` objects.

    Created by freezing an :class:`ActivationAddressGroup` via
    :meth:`ActivationAddressGroup.snapshot`.
    """

    def __init__(
        self,
        refs: List['ActivationRef'],
        layer_count: int,
        token_count: int,
    ):
        self._refs = refs
        self._layer_count = layer_count
        self._token_count = token_count

        D = ComputationalNode._HIDDEN_DIM_PLACEHOLDER
        shape: list = []
        if layer_count > 1:
            shape.append(layer_count)
        if token_count > 1:
            shape.append(token_count)
        shape.append(D)
        self._meta = torch.empty(shape, device='meta')

    def evaluate(self) -> Optional[torch.Tensor]:
        vals = [ref.evaluate() for ref in self._refs]
        if any(v is None for v in vals):
            return None
        stacked = torch.stack(vals, dim=0)
        if self._layer_count > 1 and self._token_count > 1:
            return stacked.reshape(self._layer_count, self._token_count, -1)
        return stacked

    def children(self) -> List[ComputationalNode]:
        return list(self._refs)

    def __repr__(self):
        return (
            f"RefGroup(layers={self._layer_count}, tokens={self._token_count}, "
            f"refs={len(self._refs)})"
        )


class ConstantNode(ComputationalNode):
    """
    A static number (scalar or tensor) in the graph.
    e.g., used when user does: prompt[0][5]["resid_post"] * 2.5
    """

    def __init__(self, value: Union[int, float, torch.Tensor]):
        if isinstance(value, (ActivationAddress, ActivationAddressGroup)):
            raise TypeError(
                "Activation addresses are not values. Call .snapshot() "
                "before using them in expressions or writes."
            )
        if not torch.is_tensor(value):
            self.value = torch.tensor(float(value))
        else:
            self.value = value
        self._meta = self.value.to(device='meta')

    def evaluate(self) -> torch.Tensor:
        return self.value

    def children(self) -> List[ComputationalNode]:
        return []

    def __repr__(self):
        if self.value.dim() == 0:
            return f"Const({self.value.item():.2f})"
        return f"Const(shape={list(self.value.shape)})"


# ---------------------------------------------------------------------------
# Operation nodes
# ---------------------------------------------------------------------------





class TorchFunctionNode(ComputationalNode):
    """
    Lazy node for ``torch.xxx(node, ...)`` captured via ``__torch_function__``.
    """

    def __init__(self, func: callable, args: tuple, kwargs: dict):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        # _meta is set by __torch_function__

    def evaluate(self) -> Optional[torch.Tensor]:
        resolved_args = _resolve_args(self.args)
        resolved_kwargs = _resolve_kwargs(self.kwargs)
        if any(v is None for v in resolved_args):
            return None
        return self.func(*resolved_args, **resolved_kwargs)

    def children(self) -> List[ComputationalNode]:
        kids = [a for a in self.args if isinstance(a, ComputationalNode)]
        kids.extend(v for v in self.kwargs.values() if isinstance(v, ComputationalNode))
        return kids

    def __repr__(self):
        func_name = getattr(self.func, '__name__', str(self.func))
        return f"TorchFunc({func_name})"


class AttrAccessNode(ComputationalNode):
    """
    Lazy attribute / property access on a node.

    Also callable: ``node.mean(dim=-1)`` first creates an ``AttrAccessNode``,
    then ``__call__`` produces a ``MethodCallNode``.
    """

    def __init__(self, parent: ComputationalNode, attr_name: str):
        self._parent = parent
        self._attr_name = attr_name
        # Validate that the attr exists on the meta tensor (if available)
        if parent._meta is not None:
            try:
                self._meta = getattr(parent._meta, attr_name)
            except (NotImplementedError, RuntimeError):
                self._meta = None
        else:
            self._meta = None

    def __call__(self, *args, **kwargs):
        node = MethodCallNode(self._parent, self._attr_name, args, kwargs)
        # Validate via meta tensors (if available)
        if self._parent._meta is not None:
            try:
                meta_method = getattr(self._parent._meta, self._attr_name)
                meta_args = tuple(_to_meta(a) for a in args)
                meta_kwargs = {k: _to_meta(v) for k, v in kwargs.items()}
                node._meta = meta_method(*meta_args, **meta_kwargs)
            except (NotImplementedError, RuntimeError):
                node._meta = None
        else:
            node._meta = None
        return node

    def evaluate(self):
        val = self._parent.evaluate()
        if val is None:
            return None
        return getattr(val, self._attr_name)

    def children(self) -> List[ComputationalNode]:
        return [self._parent]

    def __repr__(self):
        return f"{self._parent}.{self._attr_name}"


class MethodCallNode(ComputationalNode):
    """
    Lazy method call: ``node.method(*args, **kwargs)``.
    Created by ``AttrAccessNode.__call__``.
    """

    def __init__(
        self,
        operand: ComputationalNode,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ):
        self.operand = operand
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        # _meta is set by AttrAccessNode.__call__

    def evaluate(self) -> Optional[torch.Tensor]:
        obj = self.operand.evaluate()
        if obj is None:
            return None
        method = getattr(obj, self.method_name)
        resolved_args = _resolve_args(self.args)
        resolved_kwargs = _resolve_kwargs(self.kwargs)
        return method(*resolved_args, **resolved_kwargs)

    def children(self) -> List[ComputationalNode]:
        kids = [self.operand]
        kids.extend(a for a in self.args if isinstance(a, ComputationalNode))
        kids.extend(v for v in self.kwargs.values() if isinstance(v, ComputationalNode))
        return kids

    def __repr__(self):
        return f"{self.operand}.{self.method_name}(...)"


class IndexNode(ComputationalNode):
    """
    Lazy indexing: ``node[key]``.
    """

    def __init__(self, operand: ComputationalNode, key):
        self.operand = operand
        self.key = key
        # _meta is set by ComputationalNode.__getitem__

    def evaluate(self) -> Optional[torch.Tensor]:
        val = self.operand.evaluate()
        if val is None:
            return None
        return val[_resolve_key(self.key)]

    def children(self) -> List[ComputationalNode]:
        kids = [self.operand]
        if isinstance(self.key, ComputationalNode):
            kids.append(self.key)
        elif isinstance(self.key, tuple):
            kids.extend(k for k in self.key if isinstance(k, ComputationalNode))
        return kids

    def __repr__(self):
        return f"{self.operand}[{self.key}]"
