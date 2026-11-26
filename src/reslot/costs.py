"""Per-pick cost of every slot under each slotting objective.

``depot`` is the rectilinear distance from the depot, the proxy most re-slotting proposals use.

``aisle`` is an expected S-shape cost for one pick line stocked in the slot:

    entry(aisle) + depth_weight * depth + level_penalty * [floor or top shelf]

* ``entry`` is what the picker pays for the aisle being on the route at all: the aisle is walked
  end to end (``aisle_length``) and the front/back cross-aisle legs grow with the aisle's rank,
  ``2 * lateral_weight * x``. Under S-shape this term dominates, so it is what separates aisles.
* ``depth_weight`` is small because depth only matters when the aisle is the last of an odd number
  visited (the picker turns back at the deepest pick), which is a minority of tours.
* ``level_penalty`` keeps the fastest SKUs off floor and top-shelf slots; it costs no travel when
  enough waist-height slots exist in the same bays.

The defaults keep one aisle step (``2 * lateral_weight * aisle_spacing``) larger than everything
that varies inside an aisle (depth range plus the level penalty), so every slot of aisle ``k`` is
cheaper than every slot of aisle ``k + 1`` and the exact assignment fills aisles in order instead of
trading depth against aisle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import Layout

OBJECTIVES = ("aisle", "depot")


@dataclass(frozen=True)
class CostModel:
    lateral_weight: float = 0.5
    depth_weight: float = 0.1
    level_penalty: float = 1.5


def slot_costs(layout: Layout, objective: str, model: CostModel | None = None) -> np.ndarray:
    m = model or CostModel()
    geo = layout.geometry
    if objective == "depot":
        return layout.x + layout.y
    if objective == "aisle":
        entry = geo.aisle_length + 2.0 * m.lateral_weight * layout.x
        awkward = (layout.level == 1) | (layout.level == geo.n_levels)
        return entry + m.depth_weight * layout.y + m.level_penalty * awkward
    raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
