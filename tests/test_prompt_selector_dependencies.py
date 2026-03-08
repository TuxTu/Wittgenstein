"""
Tests for the ledger-based prompt/selector dependency refactor.

Covers:
- Selector hashability and equality
- Node registry identity: same coordinate → same ActivationRef object
- Overlapping selector accesses share node instances where coordinates overlap
- Prompt write appends to ledger; effective_writes deduplicates by key
- Triangle-rule dependency snapshot: get_affecting_writes correctness
- Snapshot freeze: later writes do not alter an already-created ref's snapshot
- LayerProxy assignment and read API (no StateNode)
- WriteRecord captured correctly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from witt.selector import IndexSelector, SliceSelector, ListSelector
from witt.computational_node import ActivationRef, ActivationRefGroup, ConstantNode, WriteRecord
from witt.prompt import Prompt, TokenProxy, LayerProxy, _sel_min, _sel_max


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_prompt(n_tokens: int = 10) -> Prompt:
    """Create a Prompt with n dummy tokens."""
    tokens = [(i, f"tok{i}") for i in range(n_tokens)]
    return Prompt(tokens=tokens)


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
    """Accessing the same coordinate twice returns the identical object."""
    p = make_prompt()
    ref1 = p[0][5]["resid_post"]
    ref2 = p[0][5]["resid_post"]
    assert ref1 is ref2, "Same coordinate must return identical ActivationRef"

    print("✓ Registry identity: same coordinate → same object")


def test_registry_identity_different_modules():
    """Different modules at the same token/layer give different refs."""
    p = make_prompt()
    ref_resid = p[0][5]["resid_post"]
    ref_mlp = p[0][5]["mlp"]
    assert ref_resid is not ref_mlp

    print("✓ Registry: different modules → different refs")


def test_registry_identity_different_layers():
    """Different layers give different refs."""
    p = make_prompt()
    ref_l5 = p[0][5]["resid_post"]
    ref_l6 = p[0][6]["resid_post"]
    assert ref_l5 is not ref_l6

    print("✓ Registry: different layers → different refs")


def test_registry_range_identity():
    """Same range selector accessed twice returns the same ref."""
    p = make_prompt()
    # Slices are concretized in Prompt.__getitem__
    ref1 = p[0:3][5]["resid_post"]
    ref2 = p[0:3][5]["resid_post"]
    assert ref1 is ref2

    print("✓ Registry: same slice access → same ref")


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
    Creating a ref after a write captures it in the snapshot.
    A subsequent write to the same position does NOT alter the already-created
    ref's snapshot.
    """
    p = make_prompt()
    v1 = ConstantNode(1.0)
    v2 = ConstantNode(2.0)

    # First write at (1, 3) — this WILL affect target (2, 5) by triangle rule
    p[1][3]["resid_post"] = v1

    # Instantiate ref at (2, 5) — snapshot should capture the write above
    ref = p[2][5]["resid_post"]
    assert len(ref.dep_snapshot) == 1
    assert ref.dep_snapshot[0].value_node is v1

    # Second write at (1, 3) with new value — ref already frozen, snapshot unchanged
    p[1][3]["resid_post"] = v2
    assert len(ref.dep_snapshot) == 1
    assert ref.dep_snapshot[0].value_node is v1, \
        "Snapshot must not change after ref instantiation"

    # A fresh read of the same coordinate returns the SAME object (registry)
    ref2 = p[2][5]["resid_post"]
    assert ref2 is ref  # Same object, same frozen snapshot

    print("✓ Snapshot frozen: later writes don't alter existing ref's dep_snapshot")


def test_snapshot_empty_when_no_affecting_writes():
    """Ref created before any writes has an empty snapshot."""
    p = make_prompt()
    ref = p[2][5]["resid_post"]
    assert ref.dep_snapshot == []

    print("✓ Snapshot empty when no writes exist at instantiation time")


def test_snapshot_excludes_non_affecting_writes():
    """
    Write at tok=6 does NOT affect target at tok=5 (6 > 5),
    so the snapshot for tok=5 is empty even if a write exists.
    """
    p = make_prompt()
    p[6][3]["resid_post"] = ConstantNode(99.0)  # tok=6, outside triangle for tok=5

    ref = p[5][3]["resid_post"]
    assert ref.dep_snapshot == [], \
        "Write at tok=6 must not appear in snapshot of target at tok=5"

    print("✓ Snapshot excludes writes outside the triangle")


# ---------------------------------------------------------------------------
# ActivationRef structure
# ---------------------------------------------------------------------------

def test_activation_ref_attributes():
    """ActivationRef stores prompt_id, token_idx, layer_idx, module, dep_snapshot."""
    p = make_prompt()
    ref = p[3][7]["mlp"]

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
    ref = p[0][5]["resid_post"]
    r = repr(ref)
    assert "P" in r
    assert "resid_post" in r
    assert "deps=" in r

    print(f"✓ ActivationRef repr: {ref!r}")


def test_activation_ref_meta_tensor():
    """Meta tensor shape is built correctly for single and multi positions."""
    p = make_prompt()

    # Single token, single layer → atomic ActivationRef with shape [D]
    ref_single = p[0][5]["resid_post"]
    assert isinstance(ref_single, ActivationRef)
    assert ref_single._meta is not None
    assert len(ref_single._meta.shape) == 1

    # Multi-token, single layer → ActivationRefGroup with shape [n_tokens, D]
    ref_multi_tok = p[0:3][5]["resid_post"]
    assert isinstance(ref_multi_tok, ActivationRefGroup)
    assert ref_multi_tok._meta is not None
    assert len(ref_multi_tok._meta.shape) == 2

    # Single token, multi-layer → ActivationRefGroup with shape [n_layers, D]
    ref_multi_layer = p[0][2:5]["resid_post"]
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
    same underlying ActivationRef objects for those positions.
    """
    p = make_prompt()

    group_a = p[0:5][3]["resid_post"]   # tokens 0,1,2,3,4
    group_b = p[3:7][3]["resid_post"]   # tokens 3,4,5,6

    assert isinstance(group_a, ActivationRefGroup)
    assert isinstance(group_b, ActivationRefGroup)

    # Tokens 3 and 4 (indices within each group) should be identical objects
    refs_a = group_a._refs  # 5 refs: tok 0..4
    refs_b = group_b._refs  # 4 refs: tok 3..6

    assert refs_a[3] is refs_b[0], "Token 3 ref must be shared"
    assert refs_a[4] is refs_b[1], "Token 4 ref must be shared"

    # Different tokens must NOT be the same
    assert refs_a[0] is not refs_b[0]

    print("✓ Overlapping selectors share atomic ActivationRefs")


def test_single_then_range_shares_ref():
    """
    A scalar access and a later range access that covers the same
    position must yield the same ActivationRef.
    """
    p = make_prompt()

    single_ref = p[2][5]["resid_post"]       # atomic ref at tok=2
    group_ref  = p[0:5][5]["resid_post"]      # group covering tok 0..4

    assert isinstance(single_ref, ActivationRef)
    assert isinstance(group_ref, ActivationRefGroup)
    assert group_ref._refs[2] is single_ref

    print("✓ Scalar access and range access share the same atomic ref")


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
    q_ref = q[2][5]["resid_post"]
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
