"""Dataset containers and CSV input/output."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .layout import SIZE_RANK, Layout, read_layout, required_size, write_layout


@dataclass(frozen=True)
class Sku:
    sku: str
    name: str
    category: str
    length_cm: float
    width_cm: float
    height_cm: float
    case_qty: int

    @property
    def unit_cube_l(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm / 1000.0

    @property
    def case_cube_l(self) -> float:
        return self.unit_cube_l * self.case_qty

    @property
    def size_need(self) -> str:
        size = required_size(self.case_cube_l)
        if size is None:
            raise ValueError(f"{self.sku}: case of {self.case_cube_l:.1f} L fits no slot size")
        return size


@dataclass
class Orders:
    """Order lines in arrival order. Lines of one order are contiguous."""

    order_ids: list[str]
    order_week: np.ndarray  # (n_orders,) 1-based week of each order
    line_order: np.ndarray  # (n_lines,) index into order_ids, non-decreasing
    line_sku: np.ndarray  # (n_lines,) index into the SKU list
    line_qty: np.ndarray  # (n_lines,)
    offsets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.offsets = np.searchsorted(self.line_order, np.arange(len(self.order_ids) + 1))

    @property
    def n_orders(self) -> int:
        return len(self.order_ids)

    @property
    def n_lines(self) -> int:
        return int(self.line_sku.shape[0])

    @property
    def n_weeks(self) -> int:
        return int(self.order_week.max()) if self.n_orders else 0

    def order_skus(self, i: int) -> list[int]:
        """Distinct SKU indices on order ``i``, in line order."""
        lo, hi = self.offsets[i], self.offsets[i + 1]
        return list(dict.fromkeys(int(s) for s in self.line_sku[lo:hi]))

    def order_indices(self, first_week: int, last_week: int) -> np.ndarray:
        mask = (self.order_week >= first_week) & (self.order_week <= last_week)
        return np.flatnonzero(mask)

    def line_mask(self, first_week: int, last_week: int) -> np.ndarray:
        weeks = self.order_week[self.line_order]
        return (weeks >= first_week) & (weeks <= last_week)


@dataclass
class Dataset:
    layout: Layout
    skus: list[Sku]
    orders: Orders
    current: np.ndarray  # slot index per SKU
    truth: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sku_index = {s.sku: i for i, s in enumerate(self.skus)}
        self.need_rank = np.array([SIZE_RANK[s.size_need] for s in self.skus], dtype=np.int64)

    @property
    def n_skus(self) -> int:
        return len(self.skus)


SKU_FIELDS = ["sku", "name", "category", "length_cm", "width_cm", "height_cm", "case_qty"]


def write_dataset(ds: Dataset, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_layout(ds.layout, out / "layout.csv")
    with (out / "skus.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SKU_FIELDS)
        for s in ds.skus:
            writer.writerow(
                [
                    s.sku,
                    s.name,
                    s.category,
                    f"{s.length_cm:.2f}",
                    f"{s.width_cm:.2f}",
                    f"{s.height_cm:.2f}",
                    s.case_qty,
                ]
            )
    o = ds.orders
    with (out / "orders.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["order_id", "week", "sku", "qty"])
        for k in range(o.n_lines):
            i = int(o.line_order[k])
            writer.writerow(
                [
                    o.order_ids[i],
                    int(o.order_week[i]),
                    ds.skus[int(o.line_sku[k])].sku,
                    int(o.line_qty[k]),
                ]
            )
    with (out / "slotting_current.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sku", "slot"])
        for s, slot in zip(ds.skus, ds.current, strict=True):
            writer.writerow([s.sku, ds.layout.slots[int(slot)].slot_id])
    (out / "truth.json").write_text(json.dumps(ds.truth, indent=2, sort_keys=True) + "\n")


def read_dataset(path: Path) -> Dataset:
    layout = read_layout(path / "layout.csv")
    with (path / "skus.csv").open(newline="") as fh:
        skus = [
            Sku(
                sku=r["sku"],
                name=r["name"],
                category=r["category"],
                length_cm=float(r["length_cm"]),
                width_cm=float(r["width_cm"]),
                height_cm=float(r["height_cm"]),
                case_qty=int(r["case_qty"]),
            )
            for r in csv.DictReader(fh)
        ]
    sku_index = {s.sku: i for i, s in enumerate(skus)}

    order_ids: list[str] = []
    order_week: list[int] = []
    line_order: list[int] = []
    line_sku: list[int] = []
    line_qty: list[int] = []
    last_id = None
    with (path / "orders.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["order_id"] != last_id:
                order_ids.append(r["order_id"])
                order_week.append(int(r["week"]))
                last_id = r["order_id"]
            line_order.append(len(order_ids) - 1)
            line_sku.append(sku_index[r["sku"]])
            line_qty.append(int(r["qty"]))
    orders = Orders(
        order_ids=order_ids,
        order_week=np.array(order_week, dtype=np.int64),
        line_order=np.array(line_order, dtype=np.int64),
        line_sku=np.array(line_sku, dtype=np.int64),
        line_qty=np.array(line_qty, dtype=np.int64),
    )

    current = np.full(len(skus), -1, dtype=np.int64)
    with (path / "slotting_current.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            current[sku_index[r["sku"]]] = layout.slot_index[r["slot"]]
    truth_path = path / "truth.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
    return Dataset(layout=layout, skus=skus, orders=orders, current=current, truth=truth)
