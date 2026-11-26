import numpy as np
import pytest

from reslot.costs import CostModel, slot_costs
from reslot.layout import build_layout


def test_depot_cost_is_rectilinear_distance():
    layout = build_layout()
    cost = slot_costs(layout, "depot")
    i = layout.slot_index["A03-R10-2"]
    assert cost[i] == pytest.approx(8.0 + 9.5)
    assert cost[layout.slot_index["A01-L01-1"]] == 0.5


def test_aisle_cost_is_aisle_major():
    layout = build_layout()
    cost = slot_costs(layout, "aisle")
    for k in range(9):
        assert cost[layout.aisle == k].max() < cost[layout.aisle == k + 1].min()


def test_aisle_cost_components():
    layout = build_layout()
    m = CostModel()
    cost = slot_costs(layout, "aisle", m)

    def at(slot_id):
        return cost[layout.slot_index[slot_id]]

    # depth matters a little, awkward levels (floor and top shelf) cost extra
    assert at("A01-L05-2") - at("A01-L01-2") == pytest.approx(m.depth_weight * 4.0)
    assert at("A01-L01-1") - at("A01-L01-2") == pytest.approx(m.level_penalty)
    assert at("A01-L01-4") - at("A01-L01-3") == pytest.approx(m.level_penalty)
    assert at("A02-L01-2") - at("A01-L01-2") == pytest.approx(2 * m.lateral_weight * 4.0)


def test_depot_objective_trades_aisle_against_depth_but_aisle_objective_does_not():
    layout = build_layout()
    depot, aisle = slot_costs(layout, "depot"), slot_costs(layout, "aisle")
    deep_front_aisle = layout.slot_index["A01-L24-2"]
    shallow_next_aisle = layout.slot_index["A02-L01-2"]
    assert depot[shallow_next_aisle] < depot[deep_front_aisle]
    assert aisle[shallow_next_aisle] > aisle[deep_front_aisle]


def test_unknown_objective():
    with pytest.raises(ValueError, match="unknown objective"):
        slot_costs(build_layout(), "cube")


def test_costs_are_float_arrays_of_slot_count():
    layout = build_layout()
    for objective in ("aisle", "depot"):
        cost = slot_costs(layout, objective)
        assert cost.shape == (len(layout),) and cost.dtype == np.float64
