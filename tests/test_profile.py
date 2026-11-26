import numpy as np
import pytest

from reslot.data import Orders
from reslot.profile import MIN_LIFT, MIN_SUPPORT, mine_affinity, profile_skus, write_profile


def _kit_pairs(ds):
    return {
        tuple(sorted((ds.sku_index[p["a"]], ds.sku_index[p["b"]]))) for p in ds.truth["kit_pairs"]
    }


def test_affinity_mining_recovers_the_kit_pairs(ds):
    found = mine_affinity(ds.orders, 1, 12)
    got = {(p.a, p.b) for p in found}
    truth = _kit_pairs(ds)
    assert len(truth) == 25
    assert len(truth & got) >= 23
    assert len(got - truth) <= 2


def test_affinity_mining_on_training_weeks_only_still_recovers_them(ds):
    got = {(p.a, p.b) for p in mine_affinity(ds.orders, 1, 8)}
    truth = _kit_pairs(ds)
    assert len(truth & got) >= 23
    assert len(got - truth) <= 2


def test_mined_pairs_respect_thresholds_and_are_sorted(ds):
    found = mine_affinity(ds.orders, 1, 12)
    assert all(p.support >= MIN_SUPPORT and p.lift >= MIN_LIFT for p in found)
    assert [p.support for p in found] == sorted((p.support for p in found), reverse=True)
    assert found == mine_affinity(ds.orders, 1, 12)


def test_mined_support_is_close_to_the_injected_support(ds):
    injected = {
        tuple(sorted((ds.sku_index[p["a"]], ds.sku_index[p["b"]]))): p["support"]
        for p in ds.truth["kit_pairs"]
    }
    for p in mine_affinity(ds.orders, 1, 12):
        assert p.support == pytest.approx(injected[(p.a, p.b)], rel=0.2)
        assert p.lift > 8


def _tiny_orders():
    # 1000 orders; SKUs 0 and 1 are together in 10 (1%); SKU 0 appears alone in 40 more and
    # SKU 1 alone in 40 more; SKU 2 is on every order and pairs with everything at lift 1.
    lines_order, lines_sku = [], []
    for i in range(1000):
        skus = [2]
        if i < 10:
            skus += [0, 1]
        elif i < 50:
            skus += [0]
        elif i < 90:
            skus += [1]
        for s in skus:
            lines_order.append(i)
            lines_sku.append(s)
    return Orders(
        order_ids=[f"o{i}" for i in range(1000)],
        order_week=np.ones(1000, dtype=np.int64),
        line_order=np.array(lines_order),
        line_sku=np.array(lines_sku),
        line_qty=np.ones(len(lines_order), dtype=np.int64),
    )


def test_affinity_support_and_lift_hand_computed():
    pairs = mine_affinity(_tiny_orders(), 1, 1)
    assert [(p.a, p.b) for p in pairs] == [(0, 1)]  # (0,2) and (1,2) have lift 1
    p = pairs[0]
    assert p.count == 10 and p.support == pytest.approx(0.01)
    assert p.lift == pytest.approx(0.01 / (0.05 * 0.05))  # 4.0


def test_affinity_below_support_threshold_is_dropped():
    assert mine_affinity(_tiny_orders(), 1, 1, min_support=0.02) == []
    assert mine_affinity(_tiny_orders(), 1, 1, min_lift=5.0) == []
    assert mine_affinity(_tiny_orders(), 2, 3) == []  # no orders in that window


def test_velocity_cube_and_cpi_match_a_direct_count(ds):
    prof = profile_skus(ds, 1, 8)
    o = ds.orders
    mask = o.line_mask(1, 8)
    top = int(np.argmax(prof.lines_per_week))
    on_sku = mask & (o.line_sku == top)
    assert prof.lines_per_week[top] == pytest.approx(on_sku.sum() / 8)
    units = o.line_qty[on_sku].sum()
    assert prof.units_per_week[top] == pytest.approx(units / 8)
    sku = ds.skus[top]
    assert prof.cube_per_week[top] == pytest.approx(units * sku.unit_cube_l / 8)
    assert prof.cpi[top] == pytest.approx(sku.case_cube_l / prof.lines_per_week[top])


def test_cpi_orders_slow_bulky_skus_after_fast_small_ones(ds):
    prof = profile_skus(ds, 1, 8)
    fast = int(np.argmax(prof.lines_per_week))
    never = np.flatnonzero(prof.lines_per_week == 0)
    assert len(never) > 0 and np.isinf(prof.cpi[never]).all()
    assert prof.cpi[fast] < np.median(prof.cpi[np.isfinite(prof.cpi)])


def test_size_eligibility_counts_slots_that_fit(ds):
    prof = profile_skus(ds, 1, 12)
    for need, expected in ((0, 1920), (1, 768), (2, 192)):
        assert (prof.eligible_slots[ds.need_rank == need] == expected).all()


def test_profile_csv_has_one_row_per_sku(ds, tmp_path):
    path = tmp_path / "profile.csv"
    write_profile(ds, profile_skus(ds, 1, 12), path)
    rows = path.read_text().splitlines()
    assert len(rows) == ds.n_skus + 1
    assert rows[0].startswith("sku,size_need,eligible_slots,lines_per_week")
