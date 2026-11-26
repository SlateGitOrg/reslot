"""SKU profiling (velocity, cube movement, cube-per-order index) and affinity mining."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import Dataset, Orders

MIN_SUPPORT = 0.005
MIN_LIFT = 3.0


@dataclass
class Profile:
    """Per-SKU demand statistics over a window of weeks. All arrays are indexed by SKU."""

    first_week: int
    last_week: int
    lines_per_week: np.ndarray  # pick lines per week: the velocity used by slot costs
    units_per_week: np.ndarray
    cube_per_week: np.ndarray  # litres of product leaving the pick face per week
    cpi: np.ndarray  # cube-per-order index: slot cube held / pick lines per week (lower = better)
    eligible_slots: np.ndarray  # number of slots each SKU can be stocked in

    @property
    def weeks(self) -> int:
        return self.last_week - self.first_week + 1


@dataclass(frozen=True)
class Pair:
    a: int
    b: int
    count: int
    support: float  # share of orders containing both
    lift: float  # support / (share with a x share with b)


def profile_skus(ds: Dataset, first_week: int, last_week: int) -> Profile:
    o = ds.orders
    weeks = last_week - first_week + 1
    mask = o.line_mask(first_week, last_week)
    skus = o.line_sku[mask]
    lines = np.bincount(skus, minlength=ds.n_skus).astype(float)
    units = np.bincount(skus, weights=o.line_qty[mask], minlength=ds.n_skus)
    unit_cube = np.array([s.unit_cube_l for s in ds.skus])
    case_cube = np.array([s.case_cube_l for s in ds.skus])
    lines_pw = lines / weeks
    with np.errstate(divide="ignore"):
        cpi = np.where(lines_pw > 0, case_cube / np.where(lines_pw > 0, lines_pw, 1.0), np.inf)
    eligible = np.array(
        [int((ds.layout.size_rank >= need).sum()) for need in ds.need_rank], dtype=np.int64
    )
    return Profile(
        first_week=first_week,
        last_week=last_week,
        lines_per_week=lines_pw,
        units_per_week=units / weeks,
        cube_per_week=units * unit_cube / weeks,
        cpi=cpi,
        eligible_slots=eligible,
    )


def mine_affinity(
    orders: Orders,
    first_week: int,
    last_week: int,
    min_support: float = MIN_SUPPORT,
    min_lift: float = MIN_LIFT,
) -> list[Pair]:
    """Pairs of SKUs ordered together at least ``min_support`` of the time with lift >= min_lift.

    Sorted by support, then lift, then SKU indices so the result is deterministic.
    """
    window = orders.order_indices(first_week, last_week)
    n = len(window)
    if n == 0:
        return []
    singles: Counter[int] = Counter()
    pairs: Counter[tuple[int, int]] = Counter()
    for i in window:
        skus = sorted(orders.order_skus(int(i)))
        singles.update(skus)
        pairs.update(combinations(skus, 2))
    found = []
    for (a, b), count in pairs.items():
        support = count / n
        if support < min_support:
            continue
        lift = support / ((singles[a] / n) * (singles[b] / n))
        if lift >= min_lift:
            found.append(Pair(a, b, count, support, lift))
    found.sort(key=lambda p: (-p.support, -p.lift, p.a, p.b))
    return found


def write_profile(ds: Dataset, prof: Profile, path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sku",
                "size_need",
                "eligible_slots",
                "lines_per_week",
                "units_per_week",
                "cube_l_per_week",
                "cpi",
            ]
        )
        for i, s in enumerate(ds.skus):
            cpi = "" if np.isinf(prof.cpi[i]) else f"{prof.cpi[i]:.3f}"
            writer.writerow(
                [
                    s.sku,
                    s.size_need,
                    int(prof.eligible_slots[i]),
                    f"{prof.lines_per_week[i]:.2f}",
                    f"{prof.units_per_week[i]:.2f}",
                    f"{prof.cube_per_week[i]:.2f}",
                    cpi,
                ]
            )
