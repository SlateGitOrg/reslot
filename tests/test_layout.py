import numpy as np

from reslot.layout import build_layout, read_layout, required_size, write_layout


def test_layout_has_1920_slots_with_exact_size_mix():
    layout = build_layout()
    assert len(layout) == 10 * 2 * 24 * 4 == 1920
    assert layout.count_by_size() == {"S": 1152, "M": 576, "L": 192}


def test_geometry_coordinates():
    layout = build_layout()
    geo = layout.geometry
    assert geo.aisle_length == 24.0 and geo.aisle_spacing == 4.0
    first = layout.slots[layout.slot_index["A01-L01-1"]]
    assert (first.x, first.y) == (0.0, 0.5)
    last = layout.slots[layout.slot_index["A10-R24-4"]]
    assert (last.x, last.y) == (36.0, 23.5)
    assert layout.y.max() < geo.aisle_length and layout.y.min() > 0


def test_slot_ids_unique_and_levels_run_one_to_four():
    layout = build_layout()
    assert len({s.slot_id for s in layout.slots}) == len(layout)
    assert set(layout.level.tolist()) == {1, 2, 3, 4}
    assert np.bincount(layout.aisle).tolist() == [192] * 10


def test_layout_is_deterministic_and_round_trips_through_csv(tmp_path):
    a, b = build_layout(), build_layout()
    assert [s.size for s in a.slots] == [s.size for s in b.slots]
    write_layout(a, tmp_path / "layout.csv")
    c = read_layout(tmp_path / "layout.csv")
    assert [(s.slot_id, s.size, s.x, s.y) for s in a.slots] == [
        (s.slot_id, s.size, s.x, s.y) for s in c.slots
    ]


def test_required_size_thresholds():
    assert required_size(15.0) == "S"
    assert required_size(15.01) == "M"
    assert required_size(40.0) == "M"
    assert required_size(120.0) == "L"
    assert required_size(120.1) is None
