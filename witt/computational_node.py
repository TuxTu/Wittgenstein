"""
Lazy computational graph for activation operations.

ComputationalNode is a generic recording proxy: any operation performed on it
(arithmetic, torch functions, method calls, indexing) is recorded as a new graph
node and replayed at evaluation time. Operations are validated at definition time
via meta tensor dry-runs.
"""
import torch
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


def _ensure_node(obj: Union['ComputationalNode', int, float, torch.Tensor]) -> 'ComputationalNode':
    """Wrap scalars/tensors into ConstantNode so arithmetic works uniformly."""
    if isinstance(obj, ComputationalNode):
        return obj
    return ConstantNode(obj)


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


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ComputationalNode:
    """
    Base class for every node in the lazy computation graph.

    Intercepts operations via three mechanisms:
      - __torch_function__  : catches torch.xxx(node, ...)
      - __getattr__         : catches node.method(...) and node.attr
      - __getitem__         : catches node[key]
    Arithmetic dunders are kept explicit for nicer __repr__.

    Every node carries a ``_meta`` tensor (device='meta', zero memory) that
    tracks the output shape.  When a new node is created the same operation
    runs on the meta tensors so shape errors surface immediately.
    """

    # Fixed placeholder for shape validation.  All activations share the same
    # hidden_dim so the actual value never causes incompatibilities.
    _HIDDEN_DIM_PLACEHOLDER: ClassVar[int] = 64

    _meta: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Generic graph traversal
    # ------------------------------------------------------------------

    def children(self) -> List['ComputationalNode']:
        """Return child nodes for graph traversal. Override in subclasses."""
        return []

    # ------------------------------------------------------------------
    # Evaluate (replays the recorded operation on real tensors)
    # ------------------------------------------------------------------

    def evaluate(self) -> Optional[torch.Tensor]:
        raise NotImplementedError

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

    # ------------------------------------------------------------------
    # Explicit arithmetic dunders  (nicer __repr__ than generic recording)
    # ------------------------------------------------------------------

    def __add__(self, other):
        node = BinaryOpNode(self, _ensure_node(other), torch.add, "+")
        node._meta = torch.add(_to_meta(self), _to_meta(other))
        return node

    def __sub__(self, other):
        node = BinaryOpNode(self, _ensure_node(other), torch.sub, "-")
        node._meta = torch.sub(_to_meta(self), _to_meta(other))
        return node

    def __mul__(self, other):
        node = BinaryOpNode(self, _ensure_node(other), torch.mul, "*")
        node._meta = torch.mul(_to_meta(self), _to_meta(other))
        return node

    def __truediv__(self, other):
        node = BinaryOpNode(self, _ensure_node(other), torch.div, "/")
        node._meta = torch.div(_to_meta(self), _to_meta(other))
        return node

    def __radd__(self, other):
        node = BinaryOpNode(_ensure_node(other), self, torch.add, "+")
        node._meta = torch.add(_to_meta(other), _to_meta(self))
        return node

    def __rsub__(self, other):
        node = BinaryOpNode(_ensure_node(other), self, torch.sub, "-")
        node._meta = torch.sub(_to_meta(other), _to_meta(self))
        return node

    def __rmul__(self, other):
        node = BinaryOpNode(_ensure_node(other), self, torch.mul, "*")
        node._meta = torch.mul(_to_meta(other), _to_meta(self))
        return node

    def __rtruediv__(self, other):
        node = BinaryOpNode(_ensure_node(other), self, torch.div, "/")
        node._meta = torch.div(_to_meta(other), _to_meta(self))
        return node

    def __neg__(self):
        node = UnaryOpNode(self, torch.neg, "-")
        node._meta = torch.neg(self._meta)
        return node

    def __abs__(self):
        node = UnaryOpNode(self, torch.abs, "abs")
        node._meta = torch.abs(self._meta)
        return node


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class ActivationRef(ComputationalNode):
    """
    A pointer to one or more activations selected by token/layer Selectors.

    This is a leaf node: it holds no computation, only an address.  The
    Executor fills in the actual tensor values via ``set_cache`` /
    ``_layer_cache`` during a forward pass.
    """

    def __init__(
        self,
        prompt_id: int,
        state_id: int,
        token_selector: Selector,
        layer_selector: Selector,
        module: str,
    ):
        self.prompt_id = prompt_id
        self.state_id = state_id
        self.token_selector = token_selector
        self.layer_selector = layer_selector
        self.module = module
        self._runtime_cache: Optional[torch.Tensor] = None
        self._layer_cache: Dict[int, torch.Tensor] = {}

        # Build meta tensor – shape is fully known at definition time
        # when selectors have concrete bounds.  For open-ended slices
        # (e.g. layer_selector = slice(None) when num_layers is unknown)
        # we fall back to _meta = None (skip shape validation).
        D = ComputationalNode._HIDDEN_DIM_PLACEHOLDER
        try:
            shape = []
            if not layer_selector.is_single:
                shape.append(layer_selector.count)
            if not token_selector.is_single:
                shape.append(token_selector.count)
            shape.append(D)
            self._meta = torch.empty(shape, device='meta')
        except ValueError:
            self._meta = None

    @property
    def key(self) -> Tuple[int, int]:
        """Lookup key for the Executor: (prompt_id, state_id)."""
        return (self.prompt_id, self.state_id)

    def evaluate(self) -> Optional[torch.Tensor]:
        if self._runtime_cache is not None:
            return self._runtime_cache
        # Multi-layer assembly
        if not self.layer_selector.is_single and self._layer_cache:
            sorted_layers = sorted(self._layer_cache.keys())
            self._runtime_cache = torch.stack(
                [self._layer_cache[l] for l in sorted_layers]
            )
            return self._runtime_cache
        return None  # Not yet filled by the executor

    def set_cache(self, activation: torch.Tensor):
        """Called by the Executor inside a hook for single-layer refs."""
        self._runtime_cache = activation

    def set_layer_cache(self, layer_idx: int, activation: torch.Tensor):
        """Called by the Executor inside a hook for multi-layer refs."""
        self._layer_cache[layer_idx] = activation

    def children(self) -> List[ComputationalNode]:
        return []

    def __repr__(self):
        return (
            f"Ref(P{self.prompt_id}.S{self.state_id}"
            f".T{self.token_selector}.L{self.layer_selector}.{self.module})"
        )


class ConstantNode(ComputationalNode):
    """
    A static number (scalar or tensor) in the graph.
    e.g., used when user does: prompt[0][5]["resid_post"] * 2.5
    """

    def __init__(self, value: Union[int, float, torch.Tensor]):
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

class BinaryOpNode(ComputationalNode):
    """
    A binary operation (add, sub, mul, div) recorded in the AST.
    """

    def __init__(
        self,
        left: ComputationalNode,
        right: ComputationalNode,
        op_func: callable,
        op_symbol: str,
    ):
        self.left = left
        self.right = right
        self.op_func = op_func
        self.op_symbol = op_symbol
        self._runtime_cache: Optional[torch.Tensor] = None
        # _meta is set by the caller (ComputationalNode.__add__ etc.)

    def evaluate(self) -> Optional[torch.Tensor]:
        if self._runtime_cache is not None:
            return self._runtime_cache
        val_l = self.left.evaluate()
        val_r = self.right.evaluate()
        if val_l is None or val_r is None:
            return None
        self._runtime_cache = self.op_func(val_l, val_r)
        return self._runtime_cache

    def children(self) -> List[ComputationalNode]:
        return [self.left, self.right]

    def __repr__(self):
        return f"({self.left} {self.op_symbol} {self.right})"


class UnaryOpNode(ComputationalNode):
    """A unary operation (__neg__, __abs__) recorded in the AST."""

    def __init__(self, operand: ComputationalNode, op_func: callable, op_symbol: str):
        self.operand = operand
        self.op_func = op_func
        self.op_symbol = op_symbol
        # _meta is set by the caller

    def evaluate(self) -> Optional[torch.Tensor]:
        val = self.operand.evaluate()
        if val is None:
            return None
        return self.op_func(val)

    def children(self) -> List[ComputationalNode]:
        return [self.operand]

    def __repr__(self):
        return f"{self.op_symbol}({self.operand})"


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
