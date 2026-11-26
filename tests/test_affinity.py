import numpy as np
import pytest

from reslot.affinity import MAX_CANDIDATES, affinity_pass
from reslot.plan import fit_target
from reslot.profile import Pair
from reslot.travel import IncrementalTravel, OrderReplay


@pytest.fixture(scope="module")
def train(ds):
    return OrderReplay(ds.orders, 1, 8, ds.layout)


@pytest.fixture(scope="module")
def aisle_target(ds, train):
    return fit_target(ds, "aisle", 8, train)


def _split_a_pair(ds, target):
    """Swap one member of a kit pair with an idle SKU in a far aisle; return slotting and pair."""
    kit = ds.truth["kit_pairs"][0]
    a, b = ds.sku_index[kit["a"]], ds.sku_index[kit["b"]]
    slot_of = target.assignment.copy()
    layout = ds.layout
    idle = [
        c
        for c in range(ds.n_skus)
        if target.picks[c] == 0
        and layout.aisle[slot_of[c]] >= 5
        and ds.need_rank[c] <= layout.size_rank[slot_of[b]]
        and ds.need_rank[b] <= layout.size_rank[slot_of[c]]
    ]
    c = idle[0]
    slot_of[b], slot_of[c] = slot_of[c], slot_of[b]
    assert layout.aisle[slot_of[a]] != layout.aisle[slot_of[b]]
    return slot_of, Pair(min(a, b), max(a, b), 200, 0.0075, 90.0), a, b


def test_incremental_swap_delta_matches_a_full_replay(ds, train, aisle_target):
    travel = IncrementalTravel(train, aisle_target.assignment)
    x, y = 5, int(np.flatnonzero(ds.layout.aisle[aisle_target.assignment] == 7)[0])
    predicted = travel.swap_delta(x, y)
    swapped = aisle_target.assignment.copy()
    swapped[x], swapped[y] = swapped[y], swapped[x]
    full = sum(train.batch_distances(swapped)) - sum(train.batch_distances(aisle_target.assignment))
    assert predicted == pytest.approx(full)
    travel.apply_swap(x, y)
    assert travel.total == pytest.approx(sum(train.batch_distances(swapped)))


def test_pass_rejoins_a_split_pair_and_travel_falls(ds, train, aisle_target):
    slot_of, pair, a, b = _split_a_pair(ds, aisle_target)
    before = sum(train.batch_distances(slot_of))
    result = affinity_pass(train, ds.layout, slot_of, ds.need_rank, aisle_target.picks, [pair])
    after = sum(train.batch_distances(result.slot_of))
    assert len(result.accepted) == 1
    assert ds.layout.aisle[result.slot_of[a]] == ds.layout.aisle[result.slot_of[b]]
    assert after < before
    assert result.accepted[0].saving == pytest.approx(before - after)


def test_pass_keeps_the_slotting_feasible(ds, train, aisle_target):
    slot_of, pair, _, _ = _split_a_pair(ds, aisle_target)
    result = affinity_pass(train, ds.layout, slot_of, ds.need_rank, aisle_target.picks, [pair])
    assert len(set(result.slot_of.tolist())) == ds.n_skus
    assert (ds.layout.size_rank[result.slot_of] >= ds.need_rank).all()


def test_candidate_cap_limits_the_search(ds, train, aisle_target):
    slot_of, pair, _, _ = _split_a_pair(ds, aisle_target)
    result = affinity_pass(
        train, ds.layout, slot_of, ds.need_rank, aisle_target.picks, [pair], max_candidates=0
    )
    assert result.candidates == 0 and result.accepted == []
    assert (result.slot_of == slot_of).all()
    assert MAX_CANDIDATES == 200


def test_pass_is_a_no_op_when_pairs_already_share_an_aisle(aisle_target):
    # every kit pair is a fast mover pair, so the aisle objective already co-locates all of them
    assert aisle_target.affinity is not None
    assert aisle_target.affinity.candidates == 0
    assert (aisle_target.slot_of == aisle_target.assignment).all()


def test_depot_objective_splits_pairs_and_the_pass_repairs_some(ds, train):
    target = fit_target(ds, "depot", 8, train)
    result = target.affinity
    assert result is not None and 0 < len(result.accepted) <= result.evaluated <= MAX_CANDIDATES
    assert all(s.saving > 0 for s in result.accepted)
    before = sum(train.batch_distances(target.assignment))
    after = sum(train.batch_distances(target.slot_of))
    assert after == pytest.approx(before - sum(s.saving for s in result.accepted))
