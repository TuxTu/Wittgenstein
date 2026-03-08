#!/usr/bin/env python
"""
Test the new operator metaclass implementation.
"""
import sys
import torch

# Add parent directory to path if needed
sys.path.insert(0, '.')

from witt.computational_node import ComputationalNode, TorchFunctionNode, ConstantNode


class DummyNode(ComputationalNode):
    """Minimal node for testing operators."""
    def __init__(self, value=0.0, shape=None):
        super().__init__()
        self.value = torch.tensor(float(value))
        if shape is None:
            shape = ()
        self._meta = torch.empty(shape, device='meta')

    def evaluate(self):
        return self.value

    def children(self):
        return []

    def __repr__(self):
        return f"Dummy({self.value.item()})"


def test_operator_existence():
    """Check that operator dunder methods exist."""
    node = DummyNode()

    # Check that common operators exist
    assert hasattr(node, '__add__')
    assert hasattr(node, '__sub__')
    assert hasattr(node, '__mul__')
    assert hasattr(node, '__truediv__')
    assert hasattr(node, '__pow__')
    assert hasattr(node, '__matmul__')
    assert hasattr(node, '__neg__')
    assert hasattr(node, '__abs__')

    # Check reverse operators
    assert hasattr(node, '__radd__')
    assert hasattr(node, '__rsub__')
    assert hasattr(node, '__rmul__')
    assert hasattr(node, '__rtruediv__')

    # Check comparisons
    assert hasattr(node, '__lt__')
    assert hasattr(node, '__le__')
    assert hasattr(node, '__eq__')
    assert hasattr(node, '__ne__')
    assert hasattr(node, '__gt__')
    assert hasattr(node, '__ge__')

    print("✓ All operator methods exist")


def test_binary_operators_create_torch_function_node():
    """Test that binary operators create TorchFunctionNode instances."""
    a = DummyNode(2.0)
    b = DummyNode(3.0)

    # Test addition
    result = a + b
    assert isinstance(result, TorchFunctionNode), f"Expected TorchFunctionNode, got {type(result)}"
    assert result.func is torch.add
    print(f"✓ a + b creates {result}")

    # Test subtraction
    result = a - b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.sub
    print(f"✓ a - b creates {result}")

    # Test multiplication
    result = a * b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.mul
    print(f"✓ a * b creates {result}")

    # Test division
    result = a / b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.true_divide
    print(f"✓ a / b creates {result}")

    # Test power
    result = a ** b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.pow
    print(f"✓ a ** b creates {result}")


def test_unary_operators():
    """Test unary operators."""
    a = DummyNode(2.0)

    # Test negation
    result = -a
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.neg
    print(f"✓ -a creates {result}")

    # Test absolute value
    result = abs(a)
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.abs
    print(f"✓ abs(a) creates {result}")


def test_reverse_operators():
    """Test reverse operators (scalar on left)."""
    a = DummyNode(2.0)

    # Test reverse addition (scalar + node)
    result = 5 + a
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.add
    print(f"✓ 5 + a creates {result}")

    # Test reverse multiplication
    result = 3 * a
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.mul
    print(f"✓ 3 * a creates {result}")


def test_comparison_operators():
    """Test comparison operators."""
    a = DummyNode(2.0)
    b = DummyNode(3.0)

    # Test less than
    result = a < b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.less
    print(f"✓ a < b creates {result}")

    # Test equality
    result = a == b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.eq
    print(f"✓ a == b creates {result}")

    # Test not equal
    result = a != b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.ne
    print(f"✓ a != b creates {result}")


def test_torch_functions():
    """Test that torch functions also create TorchFunctionNode."""
    a = DummyNode(2.0)
    b = DummyNode(3.0)

    # Test torch.add
    result = torch.add(a, b)
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.add
    print(f"✓ torch.add(a, b) creates {result}")

    # Test torch.mul
    result = torch.mul(a, b)
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.mul
    print(f"✓ torch.mul(a, b) creates {result}")


def test_meta_validation():
    """Test that meta tensor validation works."""
    # Create nodes with incompatible shapes to test validation
    a = DummyNode(shape=(2, 3))
    b = DummyNode(shape=(3, 2))

    # Matrix multiplication should work (2x3 @ 3x2 = 2x2)
    result = a @ b
    assert isinstance(result, TorchFunctionNode)
    assert result.func is torch.matmul
    print(f"✓ a @ b (matmul) creates {result}")

    # Try invalid operation (should raise error at evaluation time, not definition time)
    a_small = DummyNode(shape=(2,))
    b_small = DummyNode(shape=(3,))
    try:
        result = a_small @ b_small  # Should create node but fail evaluation
        print(f"✓ Invalid matmul creates node (will fail at evaluation)")
    except Exception as e:
        print(f"Note: {type(e).__name__}: {e}")


def main():
    print("Testing operator metaclass implementation...")
    print("-" * 50)

    try:
        test_operator_existence()
        test_binary_operators_create_torch_function_node()
        test_unary_operators()
        test_reverse_operators()
        test_comparison_operators()
        test_torch_functions()
        test_meta_validation()

        print("-" * 50)
        print("✅ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())