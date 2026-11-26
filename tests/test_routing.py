import numpy as np
import pytest

from reslot.data import Orders
from reslot.layout import build_layout
from reslot.routing import (
    largest_gap_distance,
    make_batches,
    route_distance,
    s_shape_distance,
)
from reslot.travel import OrderReplay

# Fixtures use aisle length 24 m, aisle spacing 4 m, depot at the front of aisle 1 (index 0).


def test_single_pick_goes_in_and_back_from_deepest_pick():
    assert s_shape_distance([(0, 10.0)]) == 20.0


def test_two_aisles_one_and_three():
    assert s_shape_distance([(0, 5.0), (2, 20.0)]) == 64.0


def test_three_aisles_return_last_from_deepest_pick():
    picks = [(0, 3.0), (1, 12.0), (3, 6.0), (3, 2.0)]
    assert s_shape_distance(picks) == 84.0


def test_four_aisles_even_count_traverses_the_last_aisle():
    picks = [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]
    assert s_shape_distance(picks) == 120.0


def test_first_visited_aisle_need_not_be_aisle_one():
    # depot -> aisle 3 along the front (8), in and out (10), back along the front (8)
    assert s_shape_distance([(2, 5.0)]) == 26.0


def test_no_picks_cost_nothing():
    assert s_shape_distance([]) == 0.0
    assert largest_gap_distance([]) == 0.0


def test_s_shape_ignores_pick_order_and_duplicates():
    picks = [(0, 3.0), (1, 12.0), (3, 6.0), (3, 2.0)]
    assert s_shape_distance(reversed(picks)) == s_shape_distance(picks + picks)


def test_largest_gap_single_aisle_matches_out_and_back():
    assert largest_gap_distance([(1, 4.0), (1, 9.0)]) == 2 * 4.0 + 2 * 9.0


def test_largest_gap_skips_the_biggest_gap_of_interior_aisles():
    # extremes (aisles 0 and 2) are traversed: 2*8 + 48. Aisle 1 has picks at 3 and 5 so the
    # gaps are 3, 2 and 19; the largest is skipped and the rest walked twice: 2*(24-19).
    picks = [(0, 7.0), (1, 3.0), (1, 5.0), (2, 9.0)]
    assert largest_gap_distance(picks) == 16.0 + 48.0 + 10.0


def test_largest_gap_equals_s_shape_for_two_aisles():
    picks = [(0, 5.0), (2, 20.0)]
    assert largest_gap_distance(picks) == s_shape_distance(picks) == 64.0


def test_largest_gap_beats_s_shape_for_sparse_interior_aisles():
    picks = [(0, 5.0), (1, 2.0), (2, 2.0), (3, 2.0), (4, 5.0)]
    assert largest_gap_distance(picks) < s_shape_distance(picks)


def test_route_distance_dispatch_and_unknown_policy():
    picks = [(0, 5.0), (2, 20.0)]
    assert route_distance(picks, "s-shape") == s_shape_distance(picks)
    assert route_distance(picks, "largest-gap") == largest_gap_distance(picks)
    with pytest.raises(ValueError, match="unknown routing policy"):
        route_distance(picks, "zigzag")


def test_make_batches_groups_in_arrival_order():
    batches = make_batches(list(range(20)), 8)
    assert [len(b) for b in batches] == [8, 8, 4]
    assert batches[1][0] == 8 and batches[2][-1] == 19
    with pytest.raises(ValueError):
        make_batches([1], 0)


def test_order_replay_batches_within_each_week_and_divides_by_orders():
    layout = build_layout()
    # three SKUs placed in aisle 1 depth 10.5, aisle 1 depth 0.5 and aisle 3 depth 20.5
    ids = ["A01-L11-1", "A01-L01-1", "A03-R21-1"]
    slot_of = np.array([layout.slot_index[i] for i in ids])
    # week 1: orders 0, 1, 2 (batch size 2 -> carts {0,1} and {2}); week 2: order 3 alone
    orders = Orders(
        order_ids=["o0", "o1", "o2", "o3"],
        order_week=np.array([1, 1, 1, 2]),
        line_order=np.array([0, 1, 1, 2, 3]),
        line_sku=np.array([0, 0, 1, 2, 2]),
        line_qty=np.ones(5, dtype=np.int64),
    )
    replay = OrderReplay(orders, 1, 2, layout, batch_size=2)
    assert replay.batch_orders == [2, 1, 1]
    cart_a = s_shape_distance([(0, 10.5), (0, 0.5)])  # 21.0
    cart_b = s_shape_distance([(2, 20.5)])  # 8 + 41 + 8
    assert replay.batch_distances(slot_of) == [cart_a, cart_b, cart_b]
    assert replay.per_order(slot_of) == pytest.approx((cart_a + 2 * cart_b) / 4)
