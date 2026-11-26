"""Markdown rendering of the replay, the move plan and the affinity pairs."""

from __future__ import annotations

from .data import Dataset
from .moves import Move
from .profile import Pair
from .replay import ReplayResult

HEADLINE = "full re-slot, aisle"
TOP_MOVES = 10


def _pct(x: float) -> str:
    return f"{x:+.1f}%"


def replay_table(result: ReplayResult) -> str:
    lines = [
        "| plan | SKUs moved | training m/order | held-out m/order | training vs current "
        "| held-out vs current | held-out m/order, largest-gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.rows:
        is_current = row is result.current
        lines.append(
            f"| {row.name} | {row.moves} | {row.train:.2f} | {row.held_out:.2f} "
            f"| {'-' if is_current else _pct(result.change(row, False))} "
            f"| {'-' if is_current else _pct(result.change(row, True))} "
            f"| {row.held_out_largest_gap:.2f} |"
        )
    return "\n".join(lines)


def optimism_sentence(result: ReplayResult, name: str = HEADLINE) -> str:
    row = result.row(name)
    train, held = -result.change(row, False), -result.change(row, True)
    return (
        f"`{name}` saves {train:.1f}% of travel per order on the weeks it was fitted on "
        f"(weeks 1-{result.train_weeks}) but {held:.1f}% on the held-out weeks "
        f"({result.train_weeks + 1}-{result.last_week}): an optimism gap of "
        f"{result.optimism_gap(name):.1f} points."
    )


def moves_table(ds: Dataset, moves: list[Move], limit: int = TOP_MOVES) -> str:
    ranked = sorted(moves, key=lambda m: (-m.saving, m.sku))[:limit]
    lines = [
        "| rank | sku | product | from | to | proxy saving (pick-metres/week) |",
        "|---:|---|---|---|---|---:|",
    ]
    for i, m in enumerate(ranked, 1):
        sku = ds.skus[m.sku]
        lines.append(
            f"| {i} | {sku.sku} | {sku.name} | {ds.layout.slots[m.from_slot].slot_id} "
            f"| {ds.layout.slots[m.to_slot].slot_id} | {m.saving:.1f} |"
        )
    return "\n".join(lines)


def pairs_table(ds: Dataset, pairs: list[Pair], target_slots, limit: int | None = None) -> str:
    lines = [
        "| sku a | sku b | support | lift | same aisle in aisle plan |",
        "|---|---|---:|---:|---|",
    ]
    for p in pairs[:limit]:
        same = ds.layout.aisle[target_slots[p.a]] == ds.layout.aisle[target_slots[p.b]]
        lines.append(
            f"| {ds.skus[p.a].sku} {ds.skus[p.a].name} | {ds.skus[p.b].sku} {ds.skus[p.b].name} "
            f"| {100 * p.support:.2f}% | {p.lift:.1f} | {'yes' if same else 'no'} |"
        )
    return "\n".join(lines)


def render_report(ds: Dataset, result: ReplayResult, budget: int | None = 150) -> str:
    aisle_target = result.targets["aisle"]
    swaps = aisle_target.affinity.accepted if aisle_target.affinity else []
    depot_swaps = result.targets["depot"].affinity
    parts = [
        "# Re-slotting report",
        "",
        f"Slotting is fitted on weeks 1-{result.train_weeks} ({result.n_train_orders} orders) and "
        f"scored on those weeks and on held-out weeks {result.train_weeks + 1}-{result.last_week} "
        f"({result.n_held_orders} orders). Travel is S-shape routing with carts of 8 orders in "
        "arrival order; the last column re-scores the held-out weeks under largest-gap routing.",
        "",
        "## Replay",
        "",
        replay_table(result),
        "",
        optimism_sentence(result),
        "",
        "## Top moves",
        "",
        f"The {TOP_MOVES} moves with the largest proxy saving in the "
        f"{'unlimited' if budget is None else budget}-move aisle plan "
        f"({len(result.moves[budget])} moves in all).",
        "",
        moves_table(ds, result.moves[budget]),
        "",
        "## Affinity pairs",
        "",
        f"{len(result.pairs)} pairs with support >= 0.5% and lift >= 3 in the training weeks. "
        f"The affinity pass accepted {len(swaps)} swaps for the aisle objective"
        + (f" and {len(depot_swaps.accepted)} for the depot objective." if depot_swaps else "."),
        "",
        pairs_table(ds, result.pairs, aisle_target.slot_of),
        "",
    ]
    return "\n".join(parts)
