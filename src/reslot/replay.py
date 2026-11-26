"""Held-out replay: slotting fitted on weeks 1-8, scored on weeks 1-8 and 9-12."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Dataset
from .moves import Move, apply_moves
from .plan import TRAIN_WEEKS, Target, budgeted_moves, fit_target
from .profile import Pair, mine_affinity
from .travel import OrderReplay

BUDGETS: tuple[int | None, ...] = (50, 150, 300, None)


@dataclass
class PlanScore:
    name: str
    moves: int  # SKUs relocated relative to the current slotting
    train: float  # S-shape metres per order on the training weeks
    held_out: float  # S-shape metres per order on the held-out weeks
    held_out_largest_gap: float
    slot_of: np.ndarray = field(repr=False)


@dataclass
class ReplayResult:
    train_weeks: int
    last_week: int
    n_train_orders: int
    n_held_orders: int
    rows: list[PlanScore]
    pairs: list[Pair]
    targets: dict[str, Target]
    moves: dict[int | None, list[Move]]  # aisle-objective move plans by budget

    def row(self, name: str) -> PlanScore:
        return next(r for r in self.rows if r.name == name)

    @property
    def current(self) -> PlanScore:
        return self.rows[0]

    def change(self, row: PlanScore, held_out: bool) -> float:
        """Percent change in travel per order versus the current slotting (negative = saving)."""
        base = self.current.held_out if held_out else self.current.train
        value = row.held_out if held_out else row.train
        return 100.0 * (value - base) / base

    def optimism_gap(self, name: str) -> float:
        """Training-week saving minus held-out saving, in percentage points."""
        row = self.row(name)
        return -self.change(row, False) + self.change(row, True)


def budget_name(budget: int | None) -> str:
    return "aisle, unlimited moves" if budget is None else f"aisle, {budget} moves"


def run_replay(
    ds: Dataset, train_weeks: int = TRAIN_WEEKS, budgets: tuple[int | None, ...] = BUDGETS
) -> ReplayResult:
    last_week = ds.orders.n_weeks
    if not 1 <= train_weeks < last_week:
        raise ValueError(f"train_weeks must be between 1 and {last_week - 1}")
    train = OrderReplay(ds.orders, 1, train_weeks, ds.layout)
    held = OrderReplay(ds.orders, train_weeks + 1, last_week, ds.layout)
    pairs = mine_affinity(ds.orders, 1, train_weeks)

    targets = {
        "depot": fit_target(ds, "depot", train_weeks, train, pairs),
        "aisle": fit_target(ds, "aisle", train_weeks, train, pairs),
    }

    def score(name: str, slot_of: np.ndarray) -> PlanScore:
        return PlanScore(
            name=name,
            moves=int((slot_of != ds.current).sum()),
            train=train.per_order(slot_of),
            held_out=held.per_order(slot_of),
            held_out_largest_gap=held.per_order(slot_of, "largest-gap"),
            slot_of=slot_of,
        )

    rows = [
        score("current", ds.current),
        score("full re-slot, depot", targets["depot"].slot_of),
        score("full re-slot, aisle", targets["aisle"].slot_of),
    ]
    moves: dict[int | None, list[Move]] = {}
    for budget in budgets:
        moves[budget] = budgeted_moves(ds, targets["aisle"], budget)
        rows.append(score(budget_name(budget), apply_moves(ds.current, moves[budget])))
    return ReplayResult(
        train_weeks=train_weeks,
        last_week=last_week,
        n_train_orders=train.n_orders,
        n_held_orders=held.n_orders,
        rows=rows,
        pairs=pairs,
        targets=targets,
        moves=moves,
    )
