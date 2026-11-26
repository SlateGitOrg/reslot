from itertools import permutations

import numpy as np
import pytest

from reslot.assign import assign, big_m, cost_matrix, solve, weekly_cost


def test_exact_assignment_matches_brute_force_on_6x6_and_avoids_big_m():
    rng = np.random.default_rng(3)
    picks = np.array([90.0, 55.0, 40.0, 12.0, 7.0, 1.0])
    slot_cost = np.array([1.0, 2.5, 3.0, 4.2, 6.0, 9.0])
    need_rank = np.array([2, 1, 0, 1, 0, 2])  # two L-only, two M-or-bigger, two any
    size_rank = np.array([2, 0, 1, 2, 0, 1])  # slots L, S, M, L, S, M
    rng.shuffle(slot_cost)
    matrix, penalty = cost_matrix(picks, slot_cost, need_rank, size_rank)
    assert (matrix >= penalty).sum() > 0  # the instance does contain infeasible pairs

    best = min(sum(matrix[i, p[i]] for i in range(6)) for p in permutations(range(6)))
    slots, total = solve(matrix)
    assert total == pytest.approx(best)
    assert all(matrix[i, slots[i]] < penalty for i in range(6))
    assert sorted(slots.tolist()) == list(range(6))
    assert (assign(picks, slot_cost, need_rank, size_rank) == slots).all()


def test_assignment_orders_hot_skus_to_cheap_slots():
    picks = np.array([1.0, 100.0, 10.0])
    slot_cost = np.array([5.0, 1.0, 3.0, 9.0])
    zeros = np.zeros(3, dtype=np.int64)
    slots = assign(picks, slot_cost, zeros, np.zeros(4, dtype=np.int64))
    # hottest SKU takes the cheapest slot (1), the next takes slot 2, the coldest slot 0
    assert slots.tolist() == [0, 1, 2]


def test_rectangular_assignment_uses_distinct_slots():
    picks = np.array([3.0, 2.0, 1.0])
    slots = assign(picks, np.array([4.0, 3.0, 2.0, 1.0, 5.0]), np.zeros(3, int), np.zeros(5, int))
    assert len(set(slots.tolist())) == 3


def test_infeasible_instance_raises():
    picks = np.array([5.0, 4.0])
    need = np.array([2, 2])  # two L-only SKUs, one L slot
    sizes = np.array([2, 0, 0])
    with pytest.raises(ValueError, match="no feasible assignment"):
        assign(picks, np.array([1.0, 2.0, 3.0]), need, sizes)


def test_big_m_exceeds_any_feasible_total():
    picks = np.array([10.0, 20.0, 30.0])
    cost = np.array([1.0, 2.0, 3.0])
    worst = (picks.sum()) * cost.max()
    assert big_m(picks, cost) > worst


def test_weekly_cost():
    picks = np.array([2.0, 3.0])
    assert weekly_cost(picks, np.array([1.0, 10.0, 100.0]), np.array([2, 0])) == 203.0
