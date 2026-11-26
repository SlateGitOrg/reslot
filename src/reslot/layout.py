"""Pick-area geometry: aisles x sides x bays x levels, slot sizes and coordinates.

Coordinates are metres. The depot sits at the front of aisle 1 (x = 0, y = 0). A picker walks the
centre line of an aisle and reaches both sides from it, so a slot's position is
``(aisle * aisle_spacing, (bay + 0.5) * bay_depth)``. The front cross-aisle is at y = 0 and the
back cross-aisle at y = aisle_length.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIZES = ("S", "M", "L")
SIZE_RANK = {name: rank for rank, name in enumerate(SIZES)}
# Usable volume of one slot, in litres. A SKU's case must fit in a slot to be stocked there.
SIZE_CAPACITY_L = {"S": 15.0, "M": 40.0, "L": 120.0}
SIZE_SHARE = {"S": 0.6, "M": 0.3, "L": 0.1}

LAYOUT_SEED = 20


def required_size(case_cube_l: float) -> str | None:
    """Smallest slot size whose capacity holds a case of this volume, or None if none does."""
    for name in SIZES:
        if case_cube_l <= SIZE_CAPACITY_L[name]:
            return name
    return None


@dataclass(frozen=True)
class Geometry:
    """Dimensions that drive every distance in the model."""

    n_aisles: int = 10
    n_bays: int = 24
    n_levels: int = 4
    aisle_length: float = 24.0
    aisle_spacing: float = 4.0

    @property
    def bay_depth(self) -> float:
        return self.aisle_length / self.n_bays

    def aisle_x(self, aisle: int) -> float:
        return aisle * self.aisle_spacing


@dataclass(frozen=True)
class Slot:
    index: int
    slot_id: str
    aisle: int  # 0-based; aisle 0 is the one the depot faces
    side: str  # "L" or "R"
    bay: int  # 0-based, 0 is nearest the front cross-aisle
    level: int  # 1 = floor ... n_levels = top shelf
    size: str
    x: float
    y: float


class Layout:
    """All slots plus array views used by the vectorised parts of the model."""

    def __init__(self, slots: list[Slot], geometry: Geometry | None = None):
        self.geometry = geometry or Geometry()
        self.slots = slots
        self.slot_index = {s.slot_id: s.index for s in slots}
        self.aisle = np.array([s.aisle for s in slots], dtype=np.int64)
        self.level = np.array([s.level for s in slots], dtype=np.int64)
        self.x = np.array([s.x for s in slots], dtype=float)
        self.y = np.array([s.y for s in slots], dtype=float)
        self.size_rank = np.array([SIZE_RANK[s.size] for s in slots], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.slots)

    def count_by_size(self) -> dict[str, int]:
        return {name: int((self.size_rank == SIZE_RANK[name]).sum()) for name in SIZES}


def make_slot_id(aisle: int, side: str, bay: int, level: int) -> str:
    return f"A{aisle + 1:02d}-{side}{bay + 1:02d}-{level}"


def build_layout(geometry: Geometry | None = None) -> Layout:
    """Build the standard layout; sizes are a fixed shuffle with exactly 60/30/10 S/M/L."""
    geo = geometry or Geometry()
    n = geo.n_aisles * 2 * geo.n_bays * geo.n_levels
    counts = {name: round(n * share) for name, share in SIZE_SHARE.items()}
    counts["S"] = n - counts["M"] - counts["L"]
    sizes = np.array(
        [name for name in SIZES for _ in range(counts[name])],
        dtype="<U1",
    )
    np.random.default_rng(LAYOUT_SEED).shuffle(sizes)

    slots: list[Slot] = []
    for aisle in range(geo.n_aisles):
        for side in ("L", "R"):
            for bay in range(geo.n_bays):
                for level in range(1, geo.n_levels + 1):
                    i = len(slots)
                    slots.append(
                        Slot(
                            index=i,
                            slot_id=make_slot_id(aisle, side, bay, level),
                            aisle=aisle,
                            side=side,
                            bay=bay,
                            level=level,
                            size=str(sizes[i]),
                            x=geo.aisle_x(aisle),
                            y=(bay + 0.5) * geo.bay_depth,
                        )
                    )
    return Layout(slots, geo)


LAYOUT_FIELDS = ["slot_id", "aisle", "side", "bay", "level", "size", "x_m", "y_m"]


def write_layout(layout: Layout, path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(LAYOUT_FIELDS)
        for s in layout.slots:
            writer.writerow([s.slot_id, s.aisle + 1, s.side, s.bay + 1, s.level, s.size, s.x, s.y])


def read_layout(path: Path, geometry: Geometry | None = None) -> Layout:
    slots: list[Slot] = []
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            slots.append(
                Slot(
                    index=i,
                    slot_id=row["slot_id"],
                    aisle=int(row["aisle"]) - 1,
                    side=row["side"],
                    bay=int(row["bay"]) - 1,
                    level=int(row["level"]),
                    size=row["size"],
                    x=float(row["x_m"]),
                    y=float(row["y_m"]),
                )
            )
    return Layout(slots, geometry)
