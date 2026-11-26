"""Exact SKU-to-slot assignment.

The cost of putting SKU ``i`` in slot ``j`` is ``picks[i] * slot_cost[j]``, or a big-M where the
SKU's case does not fit. The problem is a rectangular linear assignment, solved to optimality by
``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def big_m(picks: np.ndarray, slot_cost: np.ndarray) -> float:
    """A penalty larger than the cost of any feasible assignment."""
    return float((picks.sum() + 1.0) * (slot_cost.max() + 1.0) * 10.0)


def cost_matrix(
    picks: np.ndarray, slot_cost: np.ndarray, need_rank: np.ndarray, size_rank: np.ndarray
) -> tuple[np.ndarray, float]:
    """SKU x slot cost matrix and the big-M used for infeasible pairs."""
    penalty = big_m(picks, slot_cost)
    matrix = np.outer(picks, slot_cost)
    fits = need_rank[:, None] <= size_rank[None, :]
    return np.where(fits, matrix, penalty), penalty


def solve(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimal column for each row of a cost matrix with at least as many columns as rows."""
    rows, cols = linear_sum_assignment(matrix)
    return cols[np.argsort(rows)], float(matrix[rows, cols].sum())


def assign(
    picks: np.ndarray, slot_cost: np.ndarray, need_rank: np.ndarray, size_rank: np.ndarray
) -> np.ndarray:
    """Slot index per SKU minimising total picks x slot cost, respecting slot sizes."""
    matrix, penalty = cost_matrix(picks, slot_cost, need_rank, size_rank)
    slots, _ = solve(matrix)
    if (matrix[np.arange(len(slots)), slots] >= penalty).any():
        raise ValueError("no feasible assignment: some SKU only fits slots that are all taken")
    return slots


def weekly_cost(picks: np.ndarray, slot_cost: np.ndarray, slot_of: np.ndarray) -> float:
    """Proxy cost of a slotting: sum of weekly picks x slot cost."""
    return float((picks * slot_cost[slot_of]).sum())
