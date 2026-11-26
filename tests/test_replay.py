import numpy as np
import pytest

from reslot.replay import run_replay
from reslot.report import render_report
from reslot.routing import s_shape_distance


def test_aisle_objective_cuts_held_out_travel_by_at_least_15_percent(result):
    row = result.row("full re-slot, aisle")
    assert result.change(row, True) <= -15.0


def test_aisle_objective_beats_depot_objective_on_held_out_weeks(result):
    aisle, depot = result.row("full re-slot, aisle"), result.row("full re-slot, depot")
    assert aisle.held_out < depot.held_out
    assert depot.held_out < result.current.held_out  # depot is still better than today


def test_training_saving_exceeds_held_out_saving(result):
    for name in ("full re-slot, aisle", "full re-slot, depot", "aisle, unlimited moves"):
        row = result.row(name)
        assert -result.change(row, False) > -result.change(row, True)
        assert result.optimism_gap(name) > 0


def test_report_prints_both_training_and_held_out_numbers(ds, result):
    text = render_report(ds, result)
    row = result.row("full re-slot, aisle")
    assert f"{row.train:.2f}" in text and f"{row.held_out:.2f}" in text
    assert f"{-result.change(row, False):.1f}%" in text
    assert f"{-result.change(row, True):.1f}%" in text
    assert "optimism gap" in text
    for heading in ("## Replay", "## Top moves", "## Affinity pairs"):
        assert heading in text


def test_held_out_saving_is_non_decreasing_with_the_move_budget(result):
    names = [f"aisle, {b} moves" for b in (50, 150, 300)] + ["aisle, unlimited moves"]
    saved = [-result.change(result.row(n), True) for n in names]
    assert saved == sorted(saved)
    assert saved[0] > 0


def test_150_moves_capture_at_least_40_percent_of_the_unlimited_saving(result):
    s150 = -result.change(result.row("aisle, 150 moves"), True)
    unlimited = -result.change(result.row("aisle, unlimited moves"), True)
    assert s150 >= 0.4 * unlimited


def test_budgeted_plans_never_exceed_the_full_re_slot(result):
    full = result.row("full re-slot, aisle").held_out
    assert result.row("aisle, unlimited moves").held_out >= full - 1e-9


def test_every_plan_is_a_valid_slotting(ds, result):
    for row in result.rows:
        assert len(set(row.slot_of.tolist())) == ds.n_skus
        assert (ds.layout.size_rank[row.slot_of] >= ds.need_rank).all()
    assert result.current.moves == 0
    assert result.row("aisle, 150 moves").moves <= 150


def test_current_travel_matches_an_independent_recomputation(ds, result):
    o, layout = ds.orders, ds.layout
    aisle, depth = layout.aisle[ds.current], layout.y[ds.current]
    total = n_orders = 0
    for week in range(9, 13):
        idx = [int(i) for i in o.order_indices(week, week)]
        for start in range(0, len(idx), 8):
            cart = idx[start : start + 8]
            skus = {s for i in cart for s in o.order_skus(i)}
            total += s_shape_distance((int(aisle[s]), float(depth[s])) for s in skus)
            n_orders += len(cart)
    assert result.current.held_out == pytest.approx(total / n_orders)
    assert result.n_held_orders == n_orders


def test_aisle_plan_concentrates_fast_movers_by_aisle_and_depot_plan_does_not(ds, result):
    velocity = result.targets["aisle"].picks
    top = np.argsort(-velocity)[:100]
    aisle_plan = ds.layout.aisle[result.row("full re-slot, aisle").slot_of[top]]
    depot_plan = ds.layout.aisle[result.row("full re-slot, depot").slot_of[top]]
    cur = ds.layout.aisle[ds.current[top]]
    assert len(set(aisle_plan.tolist())) == 1
    assert len(set(depot_plan.tolist())) >= 3
    assert len(set(cur.tolist())) >= 8


def test_largest_gap_comparator_also_improves_under_the_aisle_plan(result):
    assert (
        result.row("full re-slot, aisle").held_out_largest_gap < result.current.held_out_largest_gap
    )
    assert result.current.held_out_largest_gap < result.current.held_out


def test_replay_is_reproducible_for_the_same_seed(ds, result):
    again = run_replay(ds)
    for a, b in zip(result.rows, again.rows, strict=True):
        assert (a.name, a.moves, a.train, a.held_out) == (b.name, b.moves, b.train, b.held_out)
    assert [(m.sku, m.from_slot, m.to_slot, m.saving) for m in again.moves[150]] == [
        (m.sku, m.from_slot, m.to_slot, m.saving) for m in result.moves[150]
    ]


def test_train_weeks_must_leave_held_out_weeks(ds):
    with pytest.raises(ValueError, match="train_weeks"):
        run_replay(ds, train_weeks=12)
