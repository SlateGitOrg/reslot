"""Move budget: turn a target slotting into a ranked, size-limited list of moves.

Starting from the current slotting, two kinds of step head towards the target:

* a single move of SKU ``s`` into its target slot when that slot is empty (1 move);
* a swap of ``s`` with the occupant ``t`` of its target slot, ``t`` going to ``s``'s old slot, when
  ``t`` fits there (2 moves).

Each step is scored by its proxy saving per move, ``picks * (cost_from - cost_to)`` summed over the
SKUs it relocates, and the best step is taken until the budget is spent or no step saves anything.
A swap can leave a displaced SKU worse off; only the net saving has to be positive.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Move:
    sku: int
    from_slot: int
    to_slot: int
    saving: float  # proxy saving of this relocation alone: weekly picks x cost difference


def plan_moves(
    current: np.ndarray,
    target: np.ndarray,
    picks: np.ndarray,
    slot_cost: np.ndarray,
    need_rank: np.ndarray,
    size_rank: np.ndarray,
    budget: int | None,
) -> list[Move]:
    """Greedy move plan; ``budget`` is the maximum number of moves (None means unlimited)."""
    slot_of = current.copy()
    occupant: dict[int, int] = {int(slot): s for s, slot in enumerate(slot_of)}
    target_owner: dict[int, int] = {int(slot): s for s, slot in enumerate(target)}
    remaining = float("inf") if budget is None else budget

    def candidate(s: int) -> tuple[float, int, int | None] | None:
        cur, tgt = int(slot_of[s]), int(target[s])
        if cur == tgt:
            return None
        gain = picks[s] * (slot_cost[cur] - slot_cost[tgt])
        other = occupant.get(tgt)
        if other is None:
            saving, cost_in_moves = gain, 1
        else:
            if need_rank[other] > size_rank[cur]:
                return None
            saving, cost_in_moves = gain + picks[other] * (slot_cost[tgt] - slot_cost[cur]), 2
        if saving <= EPS:
            return None
        return saving / cost_in_moves, cost_in_moves, other

    version = np.zeros(len(slot_of), dtype=np.int64)
    heap: list[tuple[float, int, int, int]] = []

    def push(s: int) -> None:
        version[s] += 1
        found = candidate(s)
        if found is not None:
            heapq.heappush(heap, (-found[0], s, int(version[s]), found[1]))

    for s in range(len(slot_of)):
        push(s)

    plan: list[Move] = []
    while heap and remaining > 0:
        _, s, ver, n_moves = heapq.heappop(heap)
        if ver != version[s] or n_moves > remaining:
            continue
        found = candidate(s)
        if found is None:
            continue
        _, n_moves, other = found
        cur, tgt = int(slot_of[s]), int(target[s])
        plan.append(Move(s, cur, tgt, float(picks[s] * (slot_cost[cur] - slot_cost[tgt]))))
        slot_of[s] = tgt
        occupant[tgt] = s
        if other is None:
            del occupant[cur]
        else:
            gain = picks[other] * (slot_cost[tgt] - slot_cost[cur])
            plan.append(Move(other, tgt, cur, float(gain)))
            slot_of[other] = cur
            occupant[cur] = other
        remaining -= n_moves
        touched = {s, target_owner.get(cur), target_owner.get(tgt)}
        if other is not None:
            touched.add(other)
        for t in touched:
            if t is not None:
                push(t)
    return plan


def apply_moves(current: np.ndarray, moves: list[Move]) -> np.ndarray:
    """Slotting after executing ``moves`` in order, checking each lands in a free slot or swaps."""
    slot_of = current.copy()
    for m in moves:
        if slot_of[m.sku] != m.from_slot:
            raise ValueError(f"move of SKU {m.sku} does not start from its current slot")
        slot_of[m.sku] = m.to_slot
    if len(set(slot_of.tolist())) != len(slot_of):
        raise ValueError("moves leave two SKUs in one slot")
    return slot_of
