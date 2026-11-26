"""Seeded synthetic data: SKUs, 12 weeks of orders, and a legacy-biased current slotting.

Everything here is synthetic. The properties the analysis has to recover are injected on purpose:

* Zipf popularity (exponent 1.1) over 1,500 SKUs.
* 25 kit pairs ordered together with support of 0.65-1.0% of orders and lift well above 8.
* 40 sun-care SKUs whose demand triples in weeks 9-12 only.
* 60 oversized SKUs whose cases fit only an L slot.
* A current slotting that is random except that 20% of the 100 fastest SKUs sit on the top level
  or in the last three aisles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import Dataset, Orders, Sku
from .layout import SIZE_RANK, SIZES, Layout, build_layout

BRANDS = ["Aurel", "Novara", "Lumen", "Fernhill", "Calder", "Oakmoor", "Veda", "Sorrel", "Halden"]
LINES = [
    "Argan",
    "Oat",
    "Charcoal",
    "Aloe",
    "Citrus",
    "Peppermint",
    "Rose",
    "Tea Tree",
    "Marine",
    "Cedar",
    "Shea",
    "Ginger",
    "Bamboo",
    "Lavender",
    "Mint",
]
# (first product, second product, category) for kit pairs that are bought together.
KIT_TYPES = [
    ("Shampoo", "Conditioner", "Hair care"),
    ("Cleanser", "Toner", "Skin care"),
    ("Face Wash", "Moisturiser", "Skin care"),
    ("Toothpaste", "Mouthwash", "Oral care"),
    ("Shave Gel", "Aftershave Balm", "Shaving"),
    ("Body Wash", "Body Lotion", "Body care"),
    ("Hair Mask", "Leave-in Serum", "Hair care"),
    ("Eye Cream", "Night Serum", "Skin care"),
    ("Hand Wash", "Hand Cream", "Body care"),
    ("Deodorant", "Body Spray", "Body care"),
    ("Lip Scrub", "Lip Balm", "Skin care"),
    ("Micellar Water", "Cotton Pads", "Skin care"),
    ("Dry Shampoo", "Hair Spray", "Hair care"),
]
GENERIC_PRODUCTS = [
    ("Shampoo", "Hair care"),
    ("Conditioner", "Hair care"),
    ("Hair Oil", "Hair care"),
    ("Hair Gel", "Hair care"),
    ("Face Serum", "Skin care"),
    ("Face Mask", "Skin care"),
    ("Exfoliating Scrub", "Skin care"),
    ("Night Cream", "Skin care"),
    ("Day Cream", "Skin care"),
    ("Toothpaste", "Oral care"),
    ("Floss", "Oral care"),
    ("Whitening Strips", "Oral care"),
    ("Razor Cartridges", "Shaving"),
    ("Shave Foam", "Shaving"),
    ("Body Scrub", "Body care"),
    ("Shower Gel", "Body care"),
    ("Roll-on", "Body care"),
    ("Foot Cream", "Body care"),
    ("Vitamin C", "Wellness"),
    ("Magnesium Tablets", "Wellness"),
    ("Collagen Sachets", "Wellness"),
    ("Cologne", "Fragrance"),
    ("Perfume Mist", "Fragrance"),
    ("Nail Polish", "Cosmetics"),
    ("Mascara", "Cosmetics"),
    ("Concealer", "Cosmetics"),
    ("Lip Tint", "Cosmetics"),
]
SUN_PRODUCTS = [
    "Sunscreen SPF30",
    "Sunscreen SPF50",
    "After-Sun Gel",
    "Sun Stick SPF50",
    "Tanning Oil",
    "Kids Sunscreen SPF50",
    "Sun Mist SPF30",
    "Face Sunscreen SPF50",
]
OVERSIZE_PRODUCTS = [
    "Salon Size Shampoo 5L",
    "Bulk Hand Sanitiser 5L",
    "Hair Dryer Pro",
    "Foot Spa Basin",
    "Facial Steamer",
    "Bathroom Scales",
    "Family Cotton Wool Pack",
    "Value Pack Razors",
    "Light Therapy Mask",
    "Refill Drum Body Wash 5L",
]

# Case volume ranges (litres) per size need; each stays clear of the slot capacity boundaries so
# rounding dimensions to 0.01 cm can never change a SKU's size need.
CASE_CUBE_RANGE = {"S": (1.5, 13.5), "M": (17.0, 37.0), "L": (46.0, 110.0)}
CASE_QTY_CHOICES = {"S": [6, 12, 24], "M": [4, 6, 12], "L": [1, 2, 4]}


@dataclass(frozen=True)
class GenConfig:
    seed: int = 7
    n_skus: int = 1500
    n_weeks: int = 12
    orders_per_week: int = 2500
    zipf_s: float = 1.1
    n_medium: int = 330
    n_oversized: int = 60
    n_kit_pairs: int = 25
    kit_support: tuple[float, float] = (0.0065, 0.010)
    n_sun_skus: int = 40
    sun_boost: float = 3.0
    sun_first_week: int = 9
    mean_lines_extra: float = 1.3  # lines = 1 + Poisson(1.3) on ordinary orders, mean 2.3
    legacy_top: int = 100
    legacy_share: float = 0.2


def _unique_name(base: str, used: set[str]) -> str:
    name, k = base, 2
    while name in used:
        name = f"{base} {k}"
        k += 1
    used.add(name)
    return name


def _dims_for(case_cube: float, qty: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Unit dimensions (cm) whose volume times ``qty`` equals ``case_cube`` litres."""
    unit_cm3 = case_cube * 1000.0 / qty
    base = unit_cm3 ** (1.0 / 3.0)
    ra, rb = rng.uniform(0.9, 1.5), rng.uniform(0.7, 1.1)
    return round(base * ra, 2), round(base * rb, 2), round(base / (ra * rb), 2)


def _pick_ranks(
    rng: np.random.Generator, low: int, high: int, count: int, taken: set[int]
) -> list[int]:
    pool = np.array([r for r in range(low, high) if r not in taken])
    chosen = rng.choice(pool, size=count, replace=False)
    taken.update(int(r) for r in chosen)
    return [int(r) for r in chosen]


def generate(config: GenConfig | None = None) -> Dataset:
    cfg = config or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    layout = build_layout()
    n = cfg.n_skus

    # ---- roles by popularity rank (0 = fastest) -------------------------------------------------
    taken: set[int] = set()
    kit_ranks = _pick_ranks(rng, 40, 900, 2 * cfg.n_kit_pairs, taken)
    sun_ranks = _pick_ranks(rng, 30, 500, cfg.n_sun_skus, taken)
    oversize_ranks = _pick_ranks(rng, 0, n, cfg.n_oversized, taken)
    medium_ranks = _pick_ranks(rng, 0, n, cfg.n_medium, taken)

    # SKU numbering is independent of popularity: sku i has rank_of[i].
    rank_of = rng.permutation(n)
    sku_of_rank = np.empty(n, dtype=np.int64)
    sku_of_rank[rank_of] = np.arange(n)

    need_of_rank = ["S"] * n
    for r in medium_ranks:
        need_of_rank[r] = "M"
    for r in oversize_ranks:
        need_of_rank[r] = "L"

    # ---- names and dimensions -------------------------------------------------------------------
    used: set[str] = set()
    name_of_rank: dict[int, tuple[str, str]] = {}
    kit_pairs_rank: list[tuple[int, int]] = []
    for j in range(cfg.n_kit_pairs):
        first, second, category = KIT_TYPES[j % len(KIT_TYPES)]
        brand = BRANDS[j % len(BRANDS)]
        line = LINES[(j * 4) % len(LINES)]
        ra, rb = kit_ranks[2 * j], kit_ranks[2 * j + 1]
        name_of_rank[ra] = (_unique_name(f"{brand} {line} {first}", used), category)
        name_of_rank[rb] = (_unique_name(f"{brand} {line} {second}", used), category)
        kit_pairs_rank.append((ra, rb))
    for r in sun_ranks:
        product = SUN_PRODUCTS[int(rng.integers(len(SUN_PRODUCTS)))]
        brand = BRANDS[int(rng.integers(len(BRANDS)))]
        name_of_rank[r] = (_unique_name(f"{brand} {product}", used), "Sun care")
    for r in oversize_ranks:
        product = OVERSIZE_PRODUCTS[int(rng.integers(len(OVERSIZE_PRODUCTS)))]
        brand = BRANDS[int(rng.integers(len(BRANDS)))]
        name_of_rank[r] = (_unique_name(f"{brand} {product}", used), "Bulky")
    for r in range(n):
        if r in name_of_rank:
            continue
        product, category = GENERIC_PRODUCTS[int(rng.integers(len(GENERIC_PRODUCTS)))]
        brand = BRANDS[int(rng.integers(len(BRANDS)))]
        line = LINES[int(rng.integers(len(LINES)))]
        name_of_rank[r] = (_unique_name(f"{brand} {line} {product}", used), category)

    skus: list[Sku] = []
    for i in range(n):
        r = int(rank_of[i])
        need = need_of_rank[r]
        lo, hi = CASE_CUBE_RANGE[need]
        case_cube = float(rng.uniform(lo, hi))
        qty = int(rng.choice(CASE_QTY_CHOICES[need]))
        length, width, height = _dims_for(case_cube, qty, rng)
        name, category = name_of_rank[r]
        skus.append(Sku(f"SKU{i + 1:04d}", name, category, length, width, height, qty))
    for s in skus:  # the injected size need must survive rounding
        assert s.size_need in SIZES

    # ---- orders ---------------------------------------------------------------------------------
    base_w = 1.0 / (np.arange(n) + 1.0) ** cfg.zipf_s  # by rank
    sun_idx = np.array([sku_of_rank[r] for r in sun_ranks])
    weight_by_sku = np.empty(n)
    weight_by_sku[sku_of_rank] = base_w

    supports = rng.uniform(*cfg.kit_support, size=cfg.n_kit_pairs)
    kit_pairs = [(int(sku_of_rank[a]), int(sku_of_rank[b])) for a, b in kit_pairs_rank]
    kit_cum = np.cumsum(supports)
    p_kit = float(kit_cum[-1])

    order_ids: list[str] = []
    order_week: list[int] = []
    line_order: list[int] = []
    line_sku: list[int] = []
    line_qty: list[int] = []
    for week in range(1, cfg.n_weeks + 1):
        w = weight_by_sku.copy()
        if week >= cfg.sun_first_week:
            w[sun_idx] *= cfg.sun_boost
        cdf = np.cumsum(w / w.sum())
        n_orders = int(rng.integers(cfg.orders_per_week - 100, cfg.orders_per_week + 101))
        stream = np.searchsorted(cdf, rng.random(n_orders * 6))
        pos = 0
        for _ in range(n_orders):
            kit = None
            if rng.random() < p_kit:
                kit = kit_pairs[int(np.searchsorted(kit_cum, rng.random() * p_kit))]
            if kit is not None:
                chosen = [kit[0], kit[1]]
                target = 2 + int(rng.poisson(0.3))
            else:
                chosen = []
                target = 1 + int(rng.poisson(cfg.mean_lines_extra))
            target = min(target, 10)
            while len(chosen) < target:
                if pos >= len(stream):
                    stream = np.searchsorted(cdf, rng.random(n_orders * 6))
                    pos = 0
                cand = int(stream[pos])
                pos += 1
                if cand not in chosen:
                    chosen.append(cand)
            order_ids.append(f"O{len(order_ids) + 1:06d}")
            order_week.append(week)
            for s in chosen:
                line_order.append(len(order_ids) - 1)
                line_sku.append(s)
                line_qty.append(1 + int(rng.poisson(0.5)))
    orders = Orders(
        order_ids=order_ids,
        order_week=np.array(order_week, dtype=np.int64),
        line_order=np.array(line_order, dtype=np.int64),
        line_sku=np.array(line_sku, dtype=np.int64),
        line_qty=np.array(line_qty, dtype=np.int64),
    )

    # ---- current slotting -----------------------------------------------------------------------
    need_rank = np.array([SIZE_RANK[s.size_need] for s in skus])
    top = [int(sku_of_rank[r]) for r in range(cfg.legacy_top)]
    current = _legacy_slotting(layout, need_rank, top, cfg.legacy_share, rng)

    truth = {
        "config": asdict(cfg),
        "kit_pairs": [
            {"a": skus[a].sku, "b": skus[b].sku, "support": round(float(sp), 5)}
            for (a, b), sp in zip(kit_pairs, supports, strict=True)
        ],
        "sun_care": sorted(skus[int(i)].sku for i in sun_idx),
        "oversized": sorted(skus[int(sku_of_rank[r])].sku for r in oversize_ranks),
        "top100": [skus[i].sku for i in top],
    }
    return Dataset(layout=layout, skus=skus, orders=orders, current=current, truth=truth)


def _legacy_slotting(
    layout: Layout,
    need_rank: np.ndarray,
    top_skus: list[int],
    legacy_share: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random feasible slotting, except a share of the fastest SKUs is pushed to bad locations."""
    geo = layout.geometry
    free = np.ones(len(layout), dtype=bool)
    slot_of = np.full(len(need_rank), -1, dtype=np.int64)
    legacy = rng.choice(top_skus, size=int(round(legacy_share * len(top_skus))), replace=False)
    bad_spot = (layout.level == geo.n_levels) | (layout.aisle >= geo.n_aisles - 3)

    def place(sku: int, allowed: np.ndarray) -> None:
        options = np.flatnonzero(free & allowed & (layout.size_rank >= need_rank[sku]))
        slot = int(rng.choice(options))
        slot_of[sku] = slot
        free[slot] = False

    legacy_set = {int(s) for s in legacy}
    for sku in sorted(legacy_set, key=lambda s: (-need_rank[s], s)):
        place(sku, bad_spot)
    everywhere = np.ones(len(layout), dtype=bool)
    rest = [s for s in range(len(need_rank)) if s not in legacy_set]
    for sku in sorted(rest, key=lambda s: (-need_rank[s], s)):
        place(sku, everywhere)
    return slot_of
