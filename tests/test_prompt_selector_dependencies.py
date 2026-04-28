"""
Tests for the ledger-based prompt/selector dependency refactor.

Covers:
- Selector hashability and equality
- Address registry identity: same coordinate → same ActivationAddress object
- Overlapping selector accesses share address instances where coordinates overlap
- Prompt write appends to ledger; effective_writes deduplicates by key
- Triangle-rule dependency snapshot: get_affecting_writes correctness
- Explicit snapshot freeze: later writes do not alter an already-created ref's snapshot
- ActivationView progressive indexing and read/write API
- WriteRecord captured correctly
- leaf_refs() uniform dependency traversal
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from types import SimpleNamespace
from witt.selector import IndexSelector, SliceSelector, ListSelector
from witt.computational_node import (
    ComputationalNode,
    ActivationAddress,
    ActivationAddressGroup,
    ActivationRef,
    ActivationRefGroup,
    ConstantNode,
    WriteRecord,
)
from witt.prompt import Prompt, ActivationView, _sel_min, _sel_max
from witt.tokenizer_wrapper import TokenizerWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prompt(n_tokens: int = 10) -> Prompt:
    """Create a Prompt with n dummy tokens."""
    tokens = [(i, f"tok{i}") for i in range(n_tokens)]
    return Prompt(tokens=tokens)


class DummyLayer(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("bias", bias)

    def forward(self, x):
        return x + self.bias


class DummyTransformer(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DummyLayer(torch.full((hidden_size,), float(i + 1)))
            for i in range(num_layers)
        ])


class DummyRuntimeModel(nn.Module):
    def __init__(self, hidden_size: int = 4, num_layers: int = 2, vocab_size: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = DummyTransformer(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden_size)

        with torch.no_grad():
            weights = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            self.embed.weight.copy_(weights.reshape(vocab_size, hidden_size))
            self.lm_head.weight.zero_()

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(t) for t in token_ids)


def make_runtime_executor(hidden_size: int = 4, num_layers: int = 2):
    from witt.executor import Executor
    from witt.prompt import PromptList

    prompts = PromptList()
    model = DummyRuntimeModel(hidden_size=hidden_size, num_layers=num_layers)
    exe = Executor(model, DummyTokenizer(), prompts)
    return exe, prompts


# ---------------------------------------------------------------------------
# Selector: hash and equality
# ---------------------------------------------------------------------------

def test_selector_hash_and_eq():
    """Same selector type with same value must be equal and share hash."""
    assert IndexSelector(3) == IndexSelector(3)
    assert IndexSelector(3) != IndexSelector(4)
    assert hash(IndexSelector(3)) == hash(IndexSelector(3))

    assert SliceSelector(slice(0, 5)) == SliceSelector(slice(0, 5))
    assert SliceSelector(slice(0, 5)) != SliceSelector(slice(0, 6))
    assert hash(SliceSelector(slice(0, 5))) == hash(SliceSelector(slice(0, 5)))

    assert ListSelector([1, 2, 3]) == ListSelector([1, 2, 3])
    assert ListSelector([1, 2, 3]) != ListSelector([1, 2])
    assert hash(ListSelector([1, 2, 3])) == hash(ListSelector([1, 2, 3]))

    print("✓ Selector hash and equality")


def test_selectors_usable_as_dict_keys():
    """Selectors must work as dict/set keys."""
    d = {
        IndexSelector(0): "a",
        IndexSelector(1): "b",
        SliceSelector(slice(0, 3)): "c",
        ListSelector([0, 1]): "d",
    }
    assert d[IndexSelector(0)] == "a"
    assert d[IndexSelector(1)] == "b"
    assert d[SliceSelector(slice(0, 3))] == "c"
    assert d[ListSelector([0, 1])] == "d"

    s = {IndexSelector(5), IndexSelector(5), IndexSelector(6)}
    assert len(s) == 2  # deduplication

    print("✓ Selectors usable as dict/set keys")


# ---------------------------------------------------------------------------
# sel_min / sel_max helpers
# ---------------------------------------------------------------------------

def test_sel_min_max_index():
    assert _sel_min(IndexSelector(3)) == 3
    assert _sel_max(IndexSelector(3)) == 3
    # Negative index resolved with dim_size
    assert _sel_min(IndexSelector(-1), 10) == 9
    assert _sel_max(IndexSelector(-2), 10) == 8

    print("✓ _sel_min/_sel_max IndexSelector")


def test_sel_min_max_slice():
    sel = SliceSelector(slice(2, 8))
    assert _sel_min(sel) == 2
    assert _sel_max(sel) == 7  # stop=8, step=1 → last=7

    open_sel = SliceSelector(slice(None, None))  # [:]
    assert _sel_min(open_sel) == 0
    assert _sel_max(open_sel) is None  # open-ended

    print("✓ _sel_min/_sel_max SliceSelector")


def test_sel_min_max_list():
    sel = ListSelector([3, 7, 1])
    assert _sel_min(sel) == 1
    assert _sel_max(sel) == 7

    print("✓ _sel_min/_sel_max ListSelector")


# ---------------------------------------------------------------------------
# Node registry: identity preservation
# ---------------------------------------------------------------------------

def test_registry_identity_same_coordinate():
    """Accessing the same coordinate twice returns the identical address."""
    p = make_prompt()
    addr1 = p[0][5]["resid_post"]
    addr2 = p[0][5]["resid_post"]
    assert isinstance(addr1, ActivationAddress)
    assert addr1 is addr2, "Same coordinate must return identical ActivationAddress"

    print("✓ Registry identity: same coordinate → same address")


def test_registry_identity_different_modules():
    """Different modules at the same token/layer give different addresses."""
    p = make_prompt()
    addr_resid = p[0][5]["resid_post"]
    addr_mlp = p[0][5]["mlp"]
    assert addr_resid is not addr_mlp

    print("✓ Registry: different modules → different addresses")


def test_registry_identity_different_layers():
    """Different layers give different addresses."""
    p = make_prompt()
    addr_l5 = p[0][5]["resid_post"]
    addr_l6 = p[0][6]["resid_post"]
    assert addr_l5 is not addr_l6

    print("✓ Registry: different layers → different addresses")


def test_registry_range_identity():
    """Same range selector accessed twice returns the same address group."""
    p = make_prompt()
    # Slices are concretized in Prompt.__getitem__
    addr1 = p[0:3][5]["resid_post"]
    addr2 = p[0:3][5]["resid_post"]
    assert isinstance(addr1, ActivationAddressGroup)
    assert addr1 is addr2

    print("✓ Registry: same slice access → same address group")


# ---------------------------------------------------------------------------
# Ledger: record_write and effective_writes
# ---------------------------------------------------------------------------

def test_record_write_appends_to_ledger():
    """Assignment appends a WriteRecord to _ledger."""
    p = make_prompt()
    assert len(p._ledger) == 0

    const_node = ConstantNode(1.0)
    p[0][5]["resid_post"] = const_node

    assert len(p._ledger) == 1
    wr = p._ledger[0]
    assert isinstance(wr, WriteRecord)
    assert wr.module == "resid_post"
    assert wr.value_node is const_node
    assert wr.write_id == 0

    print("✓ record_write appends WriteRecord to ledger")


def test_effective_writes_latest_wins():
    """Repeated writes to the same position keep only the last."""
    p = make_prompt()
    v1 = ConstantNode(1.0)
    v2 = ConstantNode(2.0)
    p[0][5]["resid_post"] = v1
    p[0][5]["resid_post"] = v2  # overrides v1

    effective = p.effective_writes()
    assert len(effective) == 1
    assert effective[0].value_node is v2

    print("✓ effective_writes: latest write per key wins")


def test_effective_writes_distinct_positions():
    """Writes to different positions are all preserved."""
    p = make_prompt()
    p[0][5]["resid_post"] = ConstantNode(1.0)
    p[1][5]["resid_post"] = ConstantNode(2.0)
    p[0][6]["resid_post"] = ConstantNode(3.0)

    effective = p.effective_writes()
    assert len(effective) == 3

    print("✓ effective_writes: distinct positions all preserved")


def test_implicit_constant_wrapping():
    """Raw scalar on RHS is auto-wrapped in ConstantNode."""
    p = make_prompt()
    p[0][5]["resid_post"] = 3.14  # raw float

    wr = p._ledger[0]
    assert isinstance(wr.value_node, ConstantNode)
    assert abs(wr.value_node.value.item() - 3.14) < 1e-5

    print("✓ Scalar RHS auto-wrapped in ConstantNode")


# ---------------------------------------------------------------------------
# Triangle rule: get_affecting_writes
# ---------------------------------------------------------------------------

def test_triangle_rule_single_positions():
    """
    Write at (w_tok, w_layer) affects target (t_tok, t_layer)
    iff t_tok >= w_tok AND t_layer >= w_layer.
    """
    p = make_prompt()
    v = ConstantNode(1.0)

    # Write at token=2, layer=3
    p[2][3]["resid_post"] = v

    # Target at (4, 5): 4>=2 and 5>=3 → affected
    writes = p.get_affecting_writes(IndexSelector(4), IndexSelector(5))
    assert len(writes) == 1

    # Target at (1, 5): 1 < 2 → NOT affected
    writes = p.get_affecting_writes(IndexSelector(1), IndexSelector(5))
    assert len(writes) == 0

    # Target at (4, 2): 2 < 3 → NOT affected
    writes = p.get_affecting_writes(IndexSelector(4), IndexSelector(2))
    assert len(writes) == 0

    # Target at (2, 3): 2>=2 and 3>=3 → affected (boundary)
    writes = p.get_affecting_writes(IndexSelector(2), IndexSelector(3))
    assert len(writes) == 1

    print("✓ Triangle rule: single-position correctness")


def test_triangle_rule_range_target():
    """
    For a range target the check is conservative:
    any overlap between the range and the write position counts.
    """
    p = make_prompt()
    p[5][3]["resid_post"] = ConstantNode(1.0)  # write at tok=5, layer=3

    # Slice target tok=0:8 (max=7), layer=5: 7>=5 and 5>=3 → affected
    writes = p.get_affecting_writes(
        SliceSelector(slice(0, 8)), IndexSelector(5)
    )
    assert len(writes) == 1

    # Slice target tok=0:4 (max=3), layer=5: 3 < 5 → NOT affected
    writes = p.get_affecting_writes(
        SliceSelector(slice(0, 4)), IndexSelector(5)
    )
    assert len(writes) == 0

    print("✓ Triangle rule: range target conservative check")


def test_triangle_rule_multiple_writes():
    """Only writes whose position satisfies the triangle are included."""
    p = make_prompt()
    p[1][2]["resid_post"] = ConstantNode(1.0)  # tok=1, layer=2
    p[3][4]["resid_post"] = ConstantNode(2.0)  # tok=3, layer=4
    p[7][1]["resid_post"] = ConstantNode(3.0)  # tok=7, layer=1

    # Target (5, 5): 5>=1✓ 5>=2✓; 5>=3✓ 5>=4✓; 5<7 → only first two
    writes = p.get_affecting_writes(IndexSelector(5), IndexSelector(5))
    assert len(writes) == 2
    write_ids = {wr.write_id for wr in writes}
    assert write_ids == {p._ledger[0].write_id, p._ledger[1].write_id}

    print("✓ Triangle rule: correct subset of multiple writes")


# ---------------------------------------------------------------------------
# Snapshot freeze
# ---------------------------------------------------------------------------

def test_snapshot_frozen_at_instantiation():
    """
    Creating a snapshot after a write captures it in the frozen ref.
    A subsequent write to the same position does NOT alter the already-created
    ref's snapshot, but a fresh snapshot observes the latest effective write.
    """
    p = make_prompt()
    v1 = ConstantNode(1.0)
    v2 = ConstantNode(2.0)

    # First write at (1, 3) — this WILL affect target (2, 5) by triangle rule
    p[1][3]["resid_post"] = v1

    # Create an address and snapshot it at (2, 5)
    addr = p[2][5]["resid_post"]
    ref = addr.snapshot()
    assert len(ref.dep_snapshot) == 1
    assert ref.dep_snapshot[0].value_node is v1

    # Second write at (1, 3) with new value — ref already frozen, snapshot unchanged
    p[1][3]["resid_post"] = v2
    assert len(ref.dep_snapshot) == 1
    assert ref.dep_snapshot[0].value_node is v1, \
        "Snapshot must not change after ref instantiation"

    # A fresh snapshot of the same address observes the updated effective write
    ref2 = addr.snapshot()
    assert ref2 is not ref
    assert len(ref2.dep_snapshot) == 1
    assert ref2.dep_snapshot[0].value_node is v2

    print("✓ Snapshot freeze: frozen refs stay stable while fresh snapshots update")


def test_snapshot_empty_when_no_affecting_writes():
    """Ref created before any writes has an empty snapshot."""
    p = make_prompt()
    ref = p[2][5]["resid_post"].snapshot()
    assert ref.dep_snapshot == []

    print("✓ Snapshot empty when no writes exist at instantiation time")


def test_snapshot_excludes_non_affecting_writes():
    """
    Write at tok=6 does NOT affect target at tok=5 (6 > 5),
    so the snapshot for tok=5 is empty even if a write exists.
    """
    p = make_prompt()
    p[6][3]["resid_post"] = ConstantNode(99.0)  # tok=6, outside triangle for tok=5

    ref = p[5][3]["resid_post"].snapshot()
    assert ref.dep_snapshot == [], \
        "Write at tok=6 must not appear in snapshot of target at tok=5"

    print("✓ Snapshot excludes writes outside the triangle")


# ---------------------------------------------------------------------------
# ActivationRef structure
# ---------------------------------------------------------------------------

def test_activation_ref_attributes():
    """ActivationRef stores prompt_id, token_idx, layer_idx, module, dep_snapshot."""
    p = make_prompt()
    ref = p[3][7]["mlp"].snapshot()

    assert isinstance(ref, ActivationRef)
    assert ref.prompt_id == p.uid
    assert ref.token_idx == 3
    assert ref.layer_idx == 7
    assert ref.module == "mlp"
    assert isinstance(ref.dep_snapshot, list)

    print("✓ ActivationRef has correct attributes")


def test_activation_ref_repr():
    """ActivationRef repr shows prompt_id, selectors, module, dep count."""
    p = make_prompt()
    ref = p[0][5]["resid_post"].snapshot()
    r = repr(ref)
    assert "P" in r
    assert "resid_post" in r
    assert "deps=" in r

    print(f"✓ ActivationRef repr: {ref!r}")


def test_activation_ref_meta_tensor():
    """Meta tensor shape is built correctly for single and multi positions."""
    p = make_prompt()

    # Single token, single layer → atomic ActivationRef with shape [D]
    ref_single = p[0][5]["resid_post"].snapshot()
    assert isinstance(ref_single, ActivationRef)
    assert ref_single._meta is not None
    assert len(ref_single._meta.shape) == 1

    # Multi-token, single layer → ActivationRefGroup with shape [n_tokens, D]
    ref_multi_tok = p[0:3][5]["resid_post"].snapshot()
    assert isinstance(ref_multi_tok, ActivationRefGroup)
    assert ref_multi_tok._meta is not None
    assert len(ref_multi_tok._meta.shape) == 2

    # Single token, multi-layer → ActivationRefGroup with shape [n_layers, D]
    ref_multi_layer = p[0][2:5]["resid_post"].snapshot()
    assert isinstance(ref_multi_layer, ActivationRefGroup)
    assert ref_multi_layer._meta is not None
    assert len(ref_multi_layer._meta.shape) == 2

    print("✓ Meta tensor shapes correct (atomic + group)")


# ---------------------------------------------------------------------------
# Atomic decomposition: overlapping selectors share refs
# ---------------------------------------------------------------------------

def test_overlapping_selectors_share_atomic_refs():
    """
    Two range accesses that overlap in token positions must share the
    same underlying ActivationAddress objects for those positions.
    """
    p = make_prompt()

    group_a = p[0:5][3]["resid_post"]   # tokens 0,1,2,3,4
    group_b = p[3:7][3]["resid_post"]   # tokens 3,4,5,6

    assert isinstance(group_a, ActivationAddressGroup)
    assert isinstance(group_b, ActivationAddressGroup)

    # Tokens 3 and 4 (indices within each group) should be identical objects
    addrs_a = group_a._addresses  # 5 addresses: tok 0..4
    addrs_b = group_b._addresses  # 4 addresses: tok 3..6

    assert addrs_a[3] is addrs_b[0], "Token 3 address must be shared"
    assert addrs_a[4] is addrs_b[1], "Token 4 address must be shared"

    # Different tokens must NOT be the same
    assert addrs_a[0] is not addrs_b[0]

    print("✓ Overlapping selectors share atomic ActivationAddresses")


def test_single_then_range_shares_ref():
    """
    A scalar access and a later range access that covers the same
    position must yield the same ActivationAddress.
    """
    p = make_prompt()

    single_addr = p[2][5]["resid_post"]       # atomic address at tok=2
    group_addr  = p[0:5][5]["resid_post"]     # group covering tok 0..4

    assert isinstance(single_addr, ActivationAddress)
    assert isinstance(group_addr, ActivationAddressGroup)
    assert group_addr._addresses[2] is single_addr

    print("✓ Scalar access and range access share the same atomic address")


# ---------------------------------------------------------------------------
# Cross-prompt dependency chain
# ---------------------------------------------------------------------------

def test_cross_prompt_snapshot():
    """
    Assigning a ref from prompt q as a patch on prompt p captures q's
    effective writes at ref-instantiation time, not p's writes.
    """
    p = make_prompt()
    q = make_prompt()

    # Write to q at (1, 3) — will affect q's (2, 5)
    q[1][3]["resid_post"] = ConstantNode(42.0)

    # Extract from q at (2, 5) — snapshot should capture the write above
    q_addr = q[2][5]["resid_post"]
    q_ref = q_addr.snapshot()
    assert len(q_ref.dep_snapshot) == 1

    # Assign the ref to p
    p[3][5]["resid_post"] = q_ref

    # p's write ledger should now hold a WriteRecord pointing to q_ref
    assert len(p._ledger) == 1
    assert p._ledger[0].value_node is q_ref

    # q_ref's snapshot still only has the write from before its instantiation
    q[1][3]["resid_post"] = ConstantNode(99.0)  # new write to q — too late
    assert len(q_ref.dep_snapshot) == 1
    assert q_ref.dep_snapshot[0].value_node.value.item() == 42.0

    print("✓ Cross-prompt snapshot: q_ref carries q's state at extraction time")


# ---------------------------------------------------------------------------
# ActivationView: type uniformity and progressive indexing
# ---------------------------------------------------------------------------

def test_activation_view_is_computational_node():
    """ActivationView at every level is a ComputationalNode."""
    p = make_prompt()
    tok_view = p[0]
    assert isinstance(tok_view, ComputationalNode), "p[0] must be a ComputationalNode"
    assert isinstance(tok_view, ActivationView)

    layer_view = p[0][5]
    assert isinstance(layer_view, ComputationalNode), "p[0][5] must be a ComputationalNode"
    assert isinstance(layer_view, ActivationView)

    print("✓ ActivationView is a ComputationalNode at every level")


def test_activation_view_progressive_state():
    """Token-only view has no layer; token+layer view has both."""
    p = make_prompt()
    tok_view = p[0]
    assert tok_view._layer_sel is None

    layer_view = p[0][5]
    assert layer_view._layer_sel is not None

    print("✓ ActivationView progressive state (layer_sel)")


def test_activation_view_getattr_error():
    """Attribute access on a view raises AttributeError, not AttrAccessNode."""
    p = make_prompt()
    try:
        _ = p[0].resid_post
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "ActivationView" in str(e)

    print("✓ ActivationView.__getattr__ raises AttributeError")


def test_activation_view_torch_op_error():
    """Arithmetic on views raises TypeError, not a silent TorchFunctionNode."""
    p = make_prompt()
    try:
        _ = torch.add(p[0], p[1])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "ActivationView" in str(e)

    print("✓ ActivationView blocks torch operations with clear error")


def test_activation_view_setitem_without_layer():
    """Assigning to a view with no layer selected raises TypeError."""
    p = make_prompt()
    try:
        p[0]["resid_post"] = ConstantNode(1.0)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Layer not yet selected" in str(e)

    print("✓ ActivationView.__setitem__ requires layer selection first")


def test_activation_view_setitem_non_string():
    """Assigning with a non-string module key raises TypeError."""
    p = make_prompt()
    try:
        p[0][5][0] = ConstantNode(1.0)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "str" in str(e)

    print("✓ ActivationView.__setitem__ requires string module key")


def test_activation_address_requires_snapshot_for_assignment():
    """Assigning an address directly must require an explicit snapshot."""
    p = make_prompt()
    q = make_prompt()

    try:
        p[0][5]["resid_post"] = q[1][5]["resid_post"]
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "snapshot" in str(e).lower()

    print("✓ Activation addresses must be snapshotted before assignment")


def test_activation_address_eval_error():
    """ActivationAddress.eval() raises a clear error."""
    p = make_prompt()
    addr = p[0][5]["resid_post"]

    try:
        addr.eval()
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "snapshot" in str(e).lower()

    print("✓ ActivationAddress.eval() requires snapshot() first")


# ---------------------------------------------------------------------------
# leaf_refs: uniform dependency traversal
# ---------------------------------------------------------------------------

def test_leaf_refs_basic():
    """leaf_refs() on a composed node returns all unfilled ActivationRef leaves."""
    p = make_prompt()
    ref1 = p[0][5]["resid_post"].snapshot()
    ref2 = p[1][5]["resid_post"].snapshot()
    # Compose: ref1 + ref2 → TorchFunctionNode
    composed = torch.add(ref1, ref2)

    leaves = composed.leaf_refs()
    assert len(leaves) == 2
    assert ref1 in leaves
    assert ref2 in leaves

    print("✓ leaf_refs() returns unfilled leaves from composed graph")


def test_leaf_refs_filled_excluded():
    """leaf_refs() excludes refs that already have a cached value."""
    p = make_prompt()
    ref1 = p[0][5]["resid_post"].snapshot()
    ref2 = p[1][5]["resid_post"].snapshot()

    # Fill ref1
    ref1.set_cache(torch.zeros(64))

    composed = torch.add(ref1, ref2)
    leaves = composed.leaf_refs()
    assert len(leaves) == 1
    assert leaves[0] is ref2

    print("✓ leaf_refs() excludes filled refs")


def test_leaf_refs_on_constant():
    """leaf_refs() on a ConstantNode returns empty list."""
    c = ConstantNode(42.0)
    assert c.leaf_refs() == []

    print("✓ leaf_refs() on ConstantNode is empty")


# ---------------------------------------------------------------------------
# Executor context manager (no model required — tests metadata injection)
# ---------------------------------------------------------------------------

from witt.executor import get_active_executor

def test_get_active_executor_none_outside_context():
    """get_active_executor() returns None when not in a 'with' block."""
    assert get_active_executor() is None

    print("✓ get_active_executor() returns None outside context")


def test_num_layers_class_var_default():
    """_NUM_LAYERS defaults to None."""
    assert ComputationalNode._NUM_LAYERS is None

    print("✓ _NUM_LAYERS defaults to None")


def test_prompt_generate_raises_outside_context():
    """Prompt.generate() raises RuntimeError outside executor context."""
    p = make_prompt()
    try:
        p.generate()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "active executor" in str(e).lower()

    print("✓ Prompt.generate() raises outside context")


def test_prompt_step_raises_outside_context():
    """Prompt.step() raises RuntimeError outside executor context."""
    p = make_prompt()
    try:
        p.step()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "active executor" in str(e).lower()

    print("✓ Prompt.step() raises outside context")


def test_prompt_forward_raises_outside_context():
    """Prompt.forward() raises RuntimeError outside executor context."""
    p = make_prompt()
    try:
        p.forward()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "active executor" in str(e).lower()

    print("✓ Prompt.forward() raises outside context")


def test_layer_validation_without_num_layers():
    """Layer selection works without _NUM_LAYERS (no validation)."""
    p = make_prompt()
    # Should not raise even with large index when _NUM_LAYERS is None
    view = p[0][999]
    assert view._layer_sel is not None

    print("✓ Layer selection works without _NUM_LAYERS set")


# ---------------------------------------------------------------------------
# PromptList and executor runtime behavior
# ---------------------------------------------------------------------------

def test_promptlist_positional_access_and_uid_lookup():
    """PromptList supports positional indexing while retaining explicit uid lookup."""
    from witt.prompt import PromptList

    prompts = PromptList()
    p0 = prompts.add(make_prompt(2))
    p1 = prompts.add(make_prompt(3))

    assert prompts[0] is p0
    assert prompts[-1] is p1
    assert prompts[0:2] == [p0, p1]
    assert prompts.by_uid(p0.uid) is p0
    assert p1 in prompts
    assert p0.uid in prompts

    print("✓ PromptList positional access and uid lookup both work")


def test_nested_executor_restores_metadata():
    """Nested executors restore outer ComputationalNode metadata on exit."""
    from witt.executor import Executor
    from witt.prompt import PromptList

    outer = Executor(DummyRuntimeModel(hidden_size=4, num_layers=2), DummyTokenizer(), PromptList())
    inner = Executor(DummyRuntimeModel(hidden_size=7, num_layers=3), DummyTokenizer(), PromptList())

    orig_hidden = ComputationalNode._HIDDEN_DIM_PLACEHOLDER
    orig_layers = ComputationalNode._NUM_LAYERS

    with outer:
        assert ComputationalNode._HIDDEN_DIM_PLACEHOLDER == 4
        assert ComputationalNode._NUM_LAYERS == 2
        with inner:
            assert ComputationalNode._HIDDEN_DIM_PLACEHOLDER == 7
            assert ComputationalNode._NUM_LAYERS == 3
        assert ComputationalNode._HIDDEN_DIM_PLACEHOLDER == 4
        assert ComputationalNode._NUM_LAYERS == 2

    assert ComputationalNode._HIDDEN_DIM_PLACEHOLDER == orig_hidden
    assert ComputationalNode._NUM_LAYERS == orig_layers

    print("✓ Nested executors restore outer metadata correctly")


def test_load_helpers_choose_safe_dtype():
    """CPU always uses fp32 while accelerator paths respect fp16 preference."""
    from witt.load import _select_device, _select_dtype

    assert _select_device(use_cpu=True) == "cpu"
    assert _select_dtype("cpu", use_fp16=True) == torch.float32
    assert _select_dtype("cuda", use_fp16=True) == torch.float16
    assert _select_dtype("mps", use_fp16=False) == torch.float32

    print("✓ Loader helpers choose safe dtype/device defaults")


def test_tokenizer_wrapper_handles_missing_chat_template():
    """TokenizerWrapper should allow plain tokenizers without chat templates."""
    class PlainTokenizer:
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    wrapper = TokenizerWrapper(PlainTokenizer())
    assert wrapper.has_chat_template is False
    assert wrapper.supports_thinking is False

    try:
        wrapper.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "chat template" in str(e).lower()

    print("✓ TokenizerWrapper handles tokenizers without chat templates")


def test_executor_write_exact_token_span_shape():
    """Executor applies exact [tokens, D] writes for a single layer."""
    exe, prompts = make_runtime_executor(hidden_size=4, num_layers=2)
    p = prompts.add(make_prompt(3))
    expected = torch.tensor([
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
    ])

    with exe:
        p[0:2][1]["resid_post"] = expected
        ref = p[0:2][1]["resid_post"].snapshot()
        exe.forward(p, ref)
        assert torch.equal(ref.evaluate(), expected)

    print("✓ Executor applies exact [tokens, D] writes")


def test_executor_write_exact_layer_span_shape():
    """Executor applies exact [layers, D] writes for a single token."""
    exe, prompts = make_runtime_executor(hidden_size=4, num_layers=2)
    p = prompts.add(make_prompt(3))
    expected = torch.tensor([
        [31.0, 32.0, 33.0, 34.0],
        [41.0, 42.0, 43.0, 44.0],
    ])

    with exe:
        p[1][0:2]["resid_post"] = expected
        ref = p[1][0:2]["resid_post"].snapshot()
        exe.forward(p, ref)
        assert torch.equal(ref.evaluate(), expected)

    print("✓ Executor applies exact [layers, D] writes")


def test_executor_write_exact_grid_shape():
    """Executor applies exact [layers, tokens, D] writes for a grid selection."""
    exe, prompts = make_runtime_executor(hidden_size=4, num_layers=2)
    p = prompts.add(make_prompt(3))
    expected = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        [
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
    ])

    with exe:
        p[0:2][0:2]["resid_post"] = expected
        ref = p[0:2][0:2]["resid_post"].snapshot()
        exe.forward(p, ref)
        assert torch.equal(ref.evaluate(), expected)

    print("✓ Executor applies exact [layers, tokens, D] writes")


def test_executor_write_broadcast_vector():
    """Executor broadcasts a single [D] vector across all selected positions."""
    exe, prompts = make_runtime_executor(hidden_size=4, num_layers=2)
    p = prompts.add(make_prompt(3))
    vec = torch.tensor([7.0, 8.0, 9.0, 10.0])
    expected = torch.tensor([
        [
            [7.0, 8.0, 9.0, 10.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
        [
            [7.0, 8.0, 9.0, 10.0],
            [7.0, 8.0, 9.0, 10.0],
        ],
    ])

    with exe:
        p[0:2][0:2]["resid_post"] = vec
        ref = p[0:2][0:2]["resid_post"].snapshot()
        exe.forward(p, ref)
        assert torch.equal(ref.evaluate(), expected)

    print("✓ Executor broadcasts [D] write vectors across selected positions")


def test_executor_write_invalid_shape_rejected():
    """Executor rejects ambiguous write shapes instead of broadcasting implicitly."""
    exe, prompts = make_runtime_executor(hidden_size=4, num_layers=2)
    p = prompts.add(make_prompt(3))

    with exe:
        p[0:2][0:2]["resid_post"] = torch.zeros(2, 4)
        try:
            exe.forward(p)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "exact logical shape" in str(e).lower()

    print("✓ Executor rejects ambiguous write shapes")


# ---------------------------------------------------------------------------
# eval: node evaluation
# ---------------------------------------------------------------------------

def test_node_eval_raises_outside_context():
    """ComputationalNode.eval() raises RuntimeError outside executor context."""
    c = ConstantNode(42.0)
    try:
        c.eval()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "active executor" in str(e).lower()

    print("✓ ComputationalNode.eval() raises outside context")


def test_constant_node_eval_via_executor_eval():
    """Executor.eval() on a ConstantNode returns the tensor directly."""
    from witt.executor import Executor

    class FakeModel:
        class config:
            hidden_size = 64
        device = 'cpu'
        # Fake enough for Executor.__enter__ to call num_layers
        class model:
            layers = [None] * 4

    class FakeTokenizer:
        pass

    from witt.prompt import PromptList
    exe = Executor(FakeModel(), FakeTokenizer(), PromptList())

    c = ConstantNode(7.0)
    with exe:
        result = exe.eval(c)
    assert result.item() == 7.0

    print("✓ Executor.eval() on ConstantNode returns tensor")


def test_composed_node_eval_with_constants():
    """Executor.eval() on composed expression of constants."""
    from witt.executor import Executor
    from witt.prompt import PromptList

    class FakeModel:
        class config:
            hidden_size = 64
        device = 'cpu'
        class model:
            layers = [None] * 4

    exe = Executor(FakeModel(), None, PromptList())

    a = ConstantNode(3.0)
    b = ConstantNode(4.0)
    composed = torch.add(a, b)  # TorchFunctionNode

    with exe:
        result = exe.eval(composed)
    assert abs(result.item() - 7.0) < 1e-5

    print("✓ Executor.eval() on composed constants returns correct result")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Prompt / Selector / Dependency tests")
    print("=" * 60)

    tests = [
        test_selector_hash_and_eq,
        test_selectors_usable_as_dict_keys,
        test_sel_min_max_index,
        test_sel_min_max_slice,
        test_sel_min_max_list,
        test_registry_identity_same_coordinate,
        test_registry_identity_different_modules,
        test_registry_identity_different_layers,
        test_registry_range_identity,
        test_record_write_appends_to_ledger,
        test_effective_writes_latest_wins,
        test_effective_writes_distinct_positions,
        test_implicit_constant_wrapping,
        test_triangle_rule_single_positions,
        test_triangle_rule_range_target,
        test_triangle_rule_multiple_writes,
        test_snapshot_frozen_at_instantiation,
        test_snapshot_empty_when_no_affecting_writes,
        test_snapshot_excludes_non_affecting_writes,
        test_activation_ref_attributes,
        test_activation_ref_repr,
        test_activation_ref_meta_tensor,
        test_overlapping_selectors_share_atomic_refs,
        test_single_then_range_shares_ref,
        test_cross_prompt_snapshot,
        # ActivationView tests
        test_activation_view_is_computational_node,
        test_activation_view_progressive_state,
        test_activation_view_getattr_error,
        test_activation_view_torch_op_error,
        test_activation_view_setitem_without_layer,
        test_activation_view_setitem_non_string,
        test_activation_address_requires_snapshot_for_assignment,
        test_activation_address_eval_error,
        # leaf_refs tests
        test_leaf_refs_basic,
        test_leaf_refs_filled_excluded,
        test_leaf_refs_on_constant,
        # Context manager tests (no model)
        test_get_active_executor_none_outside_context,
        test_num_layers_class_var_default,
        test_prompt_generate_raises_outside_context,
        test_prompt_step_raises_outside_context,
        test_prompt_forward_raises_outside_context,
        test_layer_validation_without_num_layers,
        test_promptlist_positional_access_and_uid_lookup,
        test_nested_executor_restores_metadata,
        test_load_helpers_choose_safe_dtype,
        test_tokenizer_wrapper_handles_missing_chat_template,
        test_executor_write_exact_token_span_shape,
        test_executor_write_exact_layer_span_shape,
        test_executor_write_exact_grid_shape,
        test_executor_write_broadcast_vector,
        test_executor_write_invalid_shape_rejected,
        # eval tests
        test_node_eval_raises_outside_context,
        test_constant_node_eval_via_executor_eval,
        test_composed_node_eval_with_constants,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n❌ {t.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    if failed == 0:
        print(f"✅ All {passed} tests passed!")
    else:
        print(f"❌ {failed} test(s) failed, {passed} passed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
