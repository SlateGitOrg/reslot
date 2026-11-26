import numpy as np

from reslot.data import read_dataset, write_dataset
from reslot.generate import GenConfig, generate
from reslot.layout import SIZE_RANK


def _slotting_is_valid(ds, slot_of):
    assert len(slot_of) == ds.n_skus
    assert len(set(slot_of.tolist())) == ds.n_skus  # no slot holds two SKUs
    assert (ds.layout.size_rank[slot_of] >= ds.need_rank).all()  # nobody in a too-small slot


def test_scale_matches_the_brief(ds):
    o = ds.orders
    assert ds.n_skus == 1500 and len(ds.layout) == 1920
    assert o.n_weeks == 12
    per_week = np.bincount(o.order_week)[1:]
    assert per_week.min() >= 2400 and per_week.max() <= 2600
    assert 2.2 < o.n_lines / o.n_orders < 2.4
    assert 65_000 < o.n_lines < 75_000


def test_every_sku_slotted_once_and_fits(ds):
    _slotting_is_valid(ds, ds.current)


def test_oversized_skus_fit_only_large_slots(ds):
    need = {s.sku: ds.need_rank[i] for i, s in enumerate(ds.skus)}
    oversized = ds.truth["oversized"]
    assert len(oversized) == 60
    assert all(need[s] == SIZE_RANK["L"] for s in oversized)
    assert int((ds.need_rank == SIZE_RANK["L"]).sum()) == 60
    assert int((ds.need_rank == SIZE_RANK["M"]).sum()) == 330
    for s in oversized:
        slot = ds.layout.slots[ds.current[ds.sku_index[s]]]
        assert slot.size == "L"


def test_popularity_is_zipf_with_exponent_near_1_1(ds):
    # fit the ordinary SKUs: kit pairs and sun-care SKUs are boosted on purpose
    o = ds.orders
    special = {ds.sku_index[s] for s in ds.truth["sun_care"]}
    for p in ds.truth["kit_pairs"]:
        special |= {ds.sku_index[p["a"]], ds.sku_index[p["b"]]}
    counts = np.bincount(o.line_sku[o.line_mask(1, 8)], minlength=ds.n_skus)
    ordinary = np.sort([c for i, c in enumerate(counts) if i not in special])[::-1]
    ranks = np.arange(1, 201)
    slope = np.polyfit(np.log(ranks), np.log(ordinary[:200]), 1)[0]
    assert -1.25 < slope < -0.95


def test_sun_care_demand_triples_only_in_the_last_four_weeks(ds):
    o = ds.orders
    sun = np.array([ds.sku_index[s] for s in ds.truth["sun_care"]])
    assert len(sun) == 40
    early = np.bincount(o.line_sku[o.line_mask(1, 8)], minlength=ds.n_skus) / 8
    late = np.bincount(o.line_sku[o.line_mask(9, 12)], minlength=ds.n_skus) / 4
    ratio = late[sun].sum() / early[sun].sum()
    others = np.ones(ds.n_skus, dtype=bool)
    others[sun] = False
    assert 2.3 < ratio < 3.3
    assert 0.9 < late[others].sum() / early[others].sum() < 1.05


def test_legacy_bias_puts_top_sellers_on_bad_spots(ds):
    geo = ds.layout.geometry
    top = [ds.sku_index[s] for s in ds.truth["top100"]]
    slot = ds.current[top]
    bad = (ds.layout.level[slot] == geo.n_levels) | (ds.layout.aisle[slot] >= geo.n_aisles - 3)
    assert bad.sum() >= 20  # 20 were forced there; others land there by chance


def test_same_seed_same_data_different_seed_different_data():
    a, b = generate(GenConfig(seed=5)), generate(GenConfig(seed=5))
    c = generate(GenConfig(seed=6))
    assert (a.orders.line_sku == b.orders.line_sku).all()
    assert (a.current == b.current).all()
    assert [s.name for s in a.skus] == [s.name for s in b.skus]
    assert not np.array_equal(a.orders.line_sku[:1000], c.orders.line_sku[:1000])


def test_dataset_round_trips_through_csv(ds, tmp_path):
    write_dataset(ds, tmp_path)
    back = read_dataset(tmp_path)
    assert (back.current == ds.current).all()
    assert (back.orders.line_sku == ds.orders.line_sku).all()
    assert (back.orders.line_qty == ds.orders.line_qty).all()
    assert (back.orders.order_week == ds.orders.order_week).all()
    assert (back.need_rank == ds.need_rank).all()
    assert back.truth["kit_pairs"] == ds.truth["kit_pairs"]
    assert (tmp_path / "orders.csv").stat().st_size < 3_000_000
