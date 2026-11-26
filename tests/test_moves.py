import numpy as np
import pytest

from reslot.assign import weekly_cost
from reslot.moves import Move, apply_moves, plan_moves
from reslot.plan import budgeted_moves


def _plan(current, target, picks, cost, need, sizes, budget):
    return plan_moves(
        np.array(current),
        np.array(target),
        np.array(picks, dtype=float),
        np.array(cost, dtype=float),
        np.array(need),
        np.array(sizes),
        budget,
    )


def test_single_moves_into_empty_slots_are_ranked_by_saving():
    # SKU 0 (10 picks) 3 -> 0 saves 10*(4-1); SKU 1 (5 picks) 2 -> 1 saves 5*(3-2)
    plan = _plan([3, 2], [0, 1], [10, 5], [1, 2, 3, 4], [0, 0], [0, 0, 0, 0], None)
    assert [(m.sku, m.from_slot, m.to_slot, m.saving) for m in plan] == [
        (0, 3, 0, 30.0),
        (1, 2, 1, 5.0),
    ]


def test_budget_of_one_takes_only_the_best_single_move():
    plan = _plan([3, 2], [0, 1], [10, 5], [1, 2, 3, 4], [0, 0], [0, 0, 0, 0], 1)
    assert [m.sku for m in plan] == [0]


def test_swap_counts_as_two_moves_and_reports_each_relocation():
    # SKU 0 (10 picks) sits in slot 1 (cost 2), SKU 1 (1 pick) in slot 0 (cost 1): swap
    args = ([1, 0], [0, 1], [10, 1], [1, 2], [0, 0], [0, 0])
    assert _plan(*args, budget=1) == []
    plan = _plan(*args, budget=2)
    assert [(m.sku, m.from_slot, m.to_slot, m.saving) for m in plan] == [
        (0, 1, 0, 10.0),
        (1, 0, 1, -1.0),
    ]
    assert sum(m.saving for m in plan) == 9.0


def test_swap_is_refused_when_the_displaced_sku_does_not_fit():
    # SKU 0 (S) wants slot 1 (M, cheap) held by SKU 1 (needs M) who cannot go to slot 0 (S)
    plan = _plan([0, 1], [1, 1], [10, 1], [3, 1], [0, 1], [0, 1], None)
    assert plan == []


def test_moves_that_do_not_save_anything_are_skipped():
    assert _plan([0, 1], [1, 0], [0, 0], [1, 2], [0, 0], [0, 0], None) == []


def test_apply_moves_validates_the_starting_slot_and_uniqueness():
    current = np.array([1, 0])
    out = apply_moves(current, [Move(0, 1, 0, 10.0), Move(1, 0, 1, -1.0)])
    assert out.tolist() == [0, 1]
    with pytest.raises(ValueError, match="current slot"):
        apply_moves(current, [Move(0, 5, 0, 1.0)])
    with pytest.raises(ValueError, match="two SKUs"):
        apply_moves(current, [Move(0, 1, 0, 1.0)])


def test_budgeted_plans_on_real_data_stay_feasible_and_within_budget(ds, result):
    target = result.targets["aisle"]
    for budget in (50, 150, 300):
        plan = budgeted_moves(ds, target, budget)
        assert 0 < len(plan) <= budget
        slot_of = apply_moves(ds.current, plan)
        assert len(set(slot_of.tolist())) == ds.n_skus
        assert (ds.layout.size_rank[slot_of] >= ds.need_rank).all()


def test_budget_plans_are_nested_and_proxy_saving_grows(ds, result):
    target = result.targets["aisle"]
    savings = []
    for budget in (50, 150, 300, None):
        plan = result.moves[budget]
        slot_of = apply_moves(ds.current, plan)
        saved = weekly_cost(target.picks, target.slot_cost, ds.current) - weekly_cost(
            target.picks, target.slot_cost, slot_of
        )
        assert sum(m.saving for m in plan) == pytest.approx(saved)
        savings.append(saved)
    assert savings == sorted(savings) and savings[0] > 0


def test_moves_are_ranked_best_first(ds, result):
    plan = result.moves[150]
    best = max(range(len(plan)), key=lambda i: plan[i].saving)
    assert best < 5  # the biggest saving is among the first steps taken
    positive = [m.saving for m in plan if m.saving > 0]
    assert sum(positive[:30]) > 0.5 * sum(positive)


def test_single_moves_into_empty_slots_lower_the_slot_cost(ds, result):
    # single moves into empty slots only ever lower a SKU's slot cost
    target = result.targets["aisle"]
    cost = target.slot_cost
    occupied = set(ds.current.tolist())
    singles = [m for m in result.moves[None] if m.to_slot not in occupied]
    assert singles
    assert all(cost[m.to_slot] < cost[m.from_slot] or target.picks[m.sku] == 0 for m in singles)
