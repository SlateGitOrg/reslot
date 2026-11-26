"""Replay orders against a slotting and measure picker travel."""

from __future__ import annotations

import numpy as np

from .data import Orders
from .layout import Layout
from .routing import make_batches, route_distance


class OrderReplay:
    """A fixed set of order batches (orders of one week form carts of ``batch_size``)."""

    def __init__(
        self,
        orders: Orders,
        first_week: int,
        last_week: int,
        layout: Layout,
        batch_size: int = 8,
    ):
        self.layout = layout
        self.batches: list[tuple[int, ...]] = []
        self.batch_orders: list[int] = []
        for week in range(first_week, last_week + 1):
            week_orders = [int(i) for i in orders.order_indices(week, week)]
            for batch in make_batches(week_orders, batch_size):
                skus: dict[int, None] = {}
                for i in batch:
                    skus.update(dict.fromkeys(orders.order_skus(i)))
                self.batches.append(tuple(skus))
                self.batch_orders.append(len(batch))
        self.n_orders = sum(self.batch_orders)
        self.batches_of: dict[int, list[int]] = {}
        for b, skus in enumerate(self.batches):
            for s in skus:
                self.batches_of.setdefault(s, []).append(b)

    def _geometry_lists(self, slot_of: np.ndarray) -> tuple[list[int], list[float]]:
        return self.layout.aisle[slot_of].tolist(), self.layout.y[slot_of].tolist()

    def batch_distances(self, slot_of: np.ndarray, policy: str = "s-shape") -> list[float]:
        aisle, depth = self._geometry_lists(slot_of)
        geo = self.layout.geometry
        return [
            route_distance(((aisle[s], depth[s]) for s in batch), policy, geo)
            for batch in self.batches
        ]

    def per_order(self, slot_of: np.ndarray, policy: str = "s-shape") -> float:
        """Mean travel per order: total batch distance divided by the orders in the carts."""
        return sum(self.batch_distances(slot_of, policy)) / self.n_orders


class IncrementalTravel:
    """Training-order travel that can be re-scored cheaply after swapping two SKUs."""

    def __init__(self, replay: OrderReplay, slot_of: np.ndarray, policy: str = "s-shape"):
        self.replay = replay
        self.policy = policy
        self.slot_of = slot_of.copy()
        self.cost = replay.batch_distances(self.slot_of, policy)

    @property
    def total(self) -> float:
        return float(sum(self.cost))

    def _affected(self, a: int, b: int) -> list[int]:
        return sorted(
            set(self.replay.batches_of.get(a, ())) | set(self.replay.batches_of.get(b, ()))
        )

    def _rescore(self, slot_of: np.ndarray, batch_ids: list[int]) -> list[float]:
        aisle, depth = self.replay._geometry_lists(slot_of)
        geo = self.replay.layout.geometry
        return [
            route_distance(((aisle[s], depth[s]) for s in self.replay.batches[b]), self.policy, geo)
            for b in batch_ids
        ]

    def swap_delta(self, a: int, b: int) -> float:
        """Change in total travel if SKUs ``a`` and ``b`` exchanged slots (negative = better)."""
        trial = self.slot_of.copy()
        trial[a], trial[b] = trial[b], trial[a]
        ids = self._affected(a, b)
        return sum(self._rescore(trial, ids)) - sum(self.cost[i] for i in ids)

    def apply_swap(self, a: int, b: int) -> None:
        self.slot_of[a], self.slot_of[b] = self.slot_of[b], self.slot_of[a]
        ids = self._affected(a, b)
        for i, c in zip(ids, self._rescore(self.slot_of, ids), strict=True):
            self.cost[i] = c
