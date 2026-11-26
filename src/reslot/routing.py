"""Picker routing policies and batch travel.

Both policies start and end at the depot (front of aisle 0) and use the front and back
cross-aisles to move between aisles. A pick is ``(aisle, depth)`` with depth measured from the front
cross-aisle.

S-shape (traversal): every aisle that holds a pick is walked end to end, alternating direction.
If the number of visited aisles is odd, the last aisle is entered from the front and left again
from its deepest pick instead of being traversed.

Largest-gap (comparator): the first and last visited aisles are traversed to close a loop around
the block; every aisle in between is entered from both ends, skipping its largest gap (gaps include
the stretch from each cross-aisle to the nearest pick).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .layout import Geometry

Pick = tuple[int, float]

POLICIES = ("s-shape", "largest-gap")


def s_shape_distance(picks: Iterable[Pick], geometry: Geometry | None = None) -> float:
    geo = geometry or Geometry()
    deepest: dict[int, float] = {}
    for aisle, depth in picks:
        if depth > deepest.get(aisle, -1.0):
            deepest[aisle] = depth
    if not deepest:
        return 0.0
    visited = sorted(deepest)
    n = len(visited)
    length = geo.aisle_length
    x = geo.aisle_x
    total = x(visited[0])  # depot to the first aisle along the front cross-aisle
    for i, aisle in enumerate(visited):
        if i < n - 1:
            total += length + (x(visited[i + 1]) - x(aisle))
        elif n % 2 == 1:
            total += 2.0 * deepest[aisle]
        else:
            total += length
    return total + x(visited[-1])  # back to the depot along the front cross-aisle


def _largest_gap(depths: Sequence[float], length: float) -> float:
    points = [0.0, *sorted(depths), length]
    return max(b - a for a, b in zip(points, points[1:], strict=False))


def largest_gap_distance(picks: Iterable[Pick], geometry: Geometry | None = None) -> float:
    geo = geometry or Geometry()
    by_aisle: dict[int, list[float]] = {}
    for aisle, depth in picks:
        by_aisle.setdefault(aisle, []).append(depth)
    if not by_aisle:
        return 0.0
    visited = sorted(by_aisle)
    length = geo.aisle_length
    first, last = visited[0], visited[-1]
    if first == last:
        return 2.0 * geo.aisle_x(first) + 2.0 * max(by_aisle[first])
    total = 2.0 * geo.aisle_x(last) + 2.0 * length
    for aisle in visited[1:-1]:
        total += 2.0 * (length - _largest_gap(by_aisle[aisle], length))
    return total


_POLICY_FUNCS = {"s-shape": s_shape_distance, "largest-gap": largest_gap_distance}


def route_distance(picks: Iterable[Pick], policy: str, geometry: Geometry | None = None) -> float:
    try:
        func = _POLICY_FUNCS[policy]
    except KeyError:
        raise ValueError(f"unknown routing policy {policy!r}; expected one of {POLICIES}") from None
    return func(picks, geometry)


def make_batches(order_ids: Sequence[int], size: int = 8) -> list[list[int]]:
    """Group orders into carts of ``size`` in arrival order; the last cart may be smaller."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [list(order_ids[i : i + size]) for i in range(0, len(order_ids), size)]
