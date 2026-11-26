"""Fit a target slotting on training weeks and plan moves towards it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .affinity import AffinityResult, affinity_pass
from .assign import assign
from .costs import CostModel, slot_costs
from .data import Dataset
from .moves import Move, plan_moves
from .profile import Pair, mine_affinity, profile_skus
from .travel import OrderReplay

TRAIN_WEEKS = 8


@dataclass
class Target:
    objective: str
    train_weeks: int
    picks: np.ndarray  # training velocity, lines per week
    slot_cost: np.ndarray
    pairs: list[Pair]
    assignment: np.ndarray  # exact assignment before the affinity pass
    affinity: AffinityResult | None
    slot_of: np.ndarray  # final target slotting


def fit_target(
    ds: Dataset,
    objective: str,
    train_weeks: int = TRAIN_WEEKS,
    train_replay: OrderReplay | None = None,
    pairs: list[Pair] | None = None,
    cost_model: CostModel | None = None,
    use_affinity: bool = True,
) -> Target:
    """Exact assignment on training-week velocity, then the bounded affinity swap pass."""
    prof = profile_skus(ds, 1, train_weeks)
    cost = slot_costs(ds.layout, objective, cost_model)
    slots = assign(prof.lines_per_week, cost, ds.need_rank, ds.layout.size_rank)
    if pairs is None:
        pairs = mine_affinity(ds.orders, 1, train_weeks)
    result = None
    final = slots
    if use_affinity and pairs:
        replay = train_replay or OrderReplay(ds.orders, 1, train_weeks, ds.layout)
        result = affinity_pass(replay, ds.layout, slots, ds.need_rank, prof.lines_per_week, pairs)
        final = result.slot_of
    return Target(objective, train_weeks, prof.lines_per_week, cost, pairs, slots, result, final)


def budgeted_moves(ds: Dataset, target: Target, budget: int | None) -> list[Move]:
    return plan_moves(
        ds.current,
        target.slot_of,
        target.picks,
        target.slot_cost,
        ds.need_rank,
        ds.layout.size_rank,
        budget,
    )
