import torch
from typing import Dict, Any, Union, Tuple

# 1. The Base Node for our Calculation Graph
class ComputationalNode:
    """
    Represents any node in the lazy calculation graph.
    It does not hold values (except constants), only logic/relationships.
    """
    def __add__(self, other):
        return BinaryOpNode(self, _ensure_node(other), torch.add, "+")
    
    def __sub__(self, other):
        return BinaryOpNode(self, _ensure_node(other), torch.sub, "-")
    
    def __mul__(self, other):
        return BinaryOpNode(self, _ensure_node(other), torch.mul, "*")

    def __truediv__(self, other):
        return BinaryOpNode(self, _ensure_node(other), torch.div, "/")

    def __radd__(self, other):
        return BinaryOpNode(_ensure_node(other), self, torch.add, "+")

    def __rsub__(self, other):
        return BinaryOpNode(_ensure_node(other), self, torch.sub, "-")

    def __rmul__(self, other):
        return BinaryOpNode(_ensure_node(other), self, torch.mul, "*")

    def __rtruediv__(self, other):
        return BinaryOpNode(_ensure_node(other), self, torch.div, "/")

    def evaluate(self) -> torch.Tensor:
        """
        The trigger. strictly strictly forbidden to call this 
        during definition time. It is only called by the Environment
        inside a model hook.
        """
        raise NotImplementedError()

# Helper to allow syntax like: prompt[0] + 5 (wraps 5 into a Node)
def _ensure_node(obj: Union['ComputationalNode', int, float, torch.Tensor]) -> 'ComputationalNode':
    if isinstance(obj, ComputationalNode):
        return obj
    return ConstantNode(obj)

# 2. A Reference to a specific activation (Leaf Node)
class ActivationRef(ComputationalNode):
    """
    A specific pointer to a future activation. 
    It is a leaf node that waits for the cache to be filled.
    """
    def __init__(self, prompt_id: int, state_id: int, token_idx: int, layer_idx: int, module: str):
        self.prompt_id = prompt_id
        self.state_id = state_id
        self.token_idx = token_idx
        self.layer_idx = layer_idx
        self.module = module

        self._runtime_cache: Optional[torch.Tensor] = None

    @property
    def key(self) -> tuple[int, int]:
        """Helper to get the lookup key for the Environment."""
        return (self.prompt_id, self.state_id)

    def evaluate(self):
        return self._runtime_cache

    def __cache__(self, activation: torch.Tensor):
        self._runtime_cache = activation

    def set_cache(self, activation: torch.Tensor):
        self.__cache__(activation)

    def __repr__(self):
        return f"Ref(P{self.prompt_id}.S{self.state_id}.T{self.token_idx}.L{self.layer_idx}.{self.module})"

# 3. A Constant (Leaf Node)
class ConstantNode(ComputationalNode):
    """
    Represents a static number (scalar) in the graph.
    e.g., used when user does: prompt[0] * 2.5
    """
    def __init__(self, value: Union[int, float, torch.Tensor]):
        if not torch.is_tensor(value):
            self.value = torch.tensor(float(value))
        else:
            self.value = value

    def evaluate(self) -> torch.Tensor:
        return self.value

    def __repr__(self):
        return f"Const({self.value.item():.2f})"

# 4. A Math Operation (Internal Node)
class BinaryOpNode(ComputationalNode):
    """
    Stores the OPERATION, not the result.
    It is an implementation of the AST (Abstract Syntax Tree).
    """
    def __init__(self, left: ComputationalNode, right: ComputationalNode, 
                 op_func: callable, op_symbol: str):
        self.left = left
        self.right = right
        self.op_func = op_func
        self.op_symbol = op_symbol

        self._runtime_cache: Optional[torch.Tensor] = None

    def evaluate(self) -> torch.Tensor:
        # This logic ONLY runs when the top-level evaluate() is triggered
        val_l = self.left.evaluate()
        val_r = self.right.evaluate()
        
        if self._runtime_cache is not None:
            return self._runtime_cache
        elif val_l is None or val_r is None:
            return None
        else:
            self._runtime_cache = self.op_func(val_l, val_r)
            return self._runtime_cache
        
    def __repr__(self):
        return f"({self.left} {self.op_symbol} {self.right})"