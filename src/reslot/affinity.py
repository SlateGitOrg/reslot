"""Bounded swap search that co-locates SKUs which are ordered together.

For every high-lift pair stocked in different aisles there are two candidate swaps: move one
member into the other's aisle by exchanging it with the slowest SKU in that aisle that fits, or the
mirror image. At most ``MAX_CANDIDATES`` candidates are tried, strongest pair first, and a swap is
kept only if the replayed training-order travel falls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import Layout
from .profile import Pair
from .travel import IncrementalTravel, OrderReplay

MAX_CANDIDATES = 200


@dataclass(frozen=True)
class AcceptedSwap:
    moved: int  # SKU brought into its partner's aisle
    displaced: int  # SKU that took its old slot
    anchor: int  # the partner the moved SKU joined
    saving: float  # training travel saved over the replayed batches, metres


@dataclass
class AffinityResult:
    slot_of: np.ndarray
    accepted: list[AcceptedSwap]
    candidates: int  # candidate swaps considered
    evaluated: int  # candidates that were actually replayed


def _best_partner_slot_holder(
    layout: Layout,
    slot_of: np.ndarray,
    need_rank: np.ndarray,
    picks: np.ndarray,
    mover: int,
    anchor: int,
) -> int | None:
    """Slowest SKU in the anchor's aisle that can trade slots with ``mover``."""
    aisle = layout.aisle[slot_of[anchor]]
    mover_size = layout.size_rank[slot_of[mover]]
    best: int | None = None
    for c in np.flatnonzero(layout.aisle[slot_of] == aisle):
        c = int(c)
        if c in (mover, anchor):
            continue
        if need_rank[c] > mover_size or need_rank[mover] > layout.size_rank[slot_of[c]]:
            continue
        if best is None or picks[c] < picks[best]:
            best = c
    return best


def affinity_pass(
    replay: OrderReplay,
    layout: Layout,
    slot_of: np.ndarray,
    need_rank: np.ndarray,
    picks: np.ndarray,
    pairs: list[Pair],
    max_candidates: int = MAX_CANDIDATES,
) -> AffinityResult:
    travel = IncrementalTravel(replay, slot_of)
    aisle_of = layout.aisle[slot_of]
    candidates: list[tuple[int, int, Pair]] = []
    for pair in pairs:
        if aisle_of[pair.a] == aisle_of[pair.b]:
            continue
        candidates.append((pair.b, pair.a, pair))  # move b to a's aisle
        candidates.append((pair.a, pair.b, pair))  # move a to b's aisle
    candidates = candidates[:max_candidates]

    accepted: list[AcceptedSwap] = []
    evaluated = 0
    for mover, anchor, _pair in candidates:
        if layout.aisle[travel.slot_of[mover]] == layout.aisle[travel.slot_of[anchor]]:
            continue
        other = _best_partner_slot_holder(layout, travel.slot_of, need_rank, picks, mover, anchor)
        if other is None:
            continue
        evaluated += 1
        delta = travel.swap_delta(mover, other)
        if delta < -1e-9:
            travel.apply_swap(mover, other)
            accepted.append(AcceptedSwap(mover, other, anchor, -delta))
    return AffinityResult(travel.slot_of, accepted, len(candidates), evaluated)
