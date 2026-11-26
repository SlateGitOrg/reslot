"""Command line interface: generate, profile, plan, replay, report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .costs import OBJECTIVES
from .data import Dataset, read_dataset, write_dataset
from .generate import GenConfig, generate
from .plan import TRAIN_WEEKS, budgeted_moves, fit_target
from .profile import mine_affinity, profile_skus, write_profile
from .replay import run_replay
from .report import optimism_sentence, render_report, replay_table

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

DataOpt = Annotated[Path, typer.Option("--data", help="Directory written by `generate`.")]
OutOpt = Annotated[Path, typer.Option("--out", help="Directory for results.")]
TrainOpt = Annotated[int, typer.Option("--train-weeks", min=1, help="Weeks used to fit slotting.")]


def _load(data: Path) -> Dataset:
    if not (data / "orders.csv").exists():
        raise typer.BadParameter(f"no dataset in {data}; run `reslot generate --out {data}` first")
    return read_dataset(data)


def _parse_budget(text: str) -> int | None:
    if text.lower() in {"all", "unlimited"}:
        return None
    try:
        value = int(text)
    except ValueError:
        raise typer.BadParameter("--moves takes a positive integer or 'unlimited'") from None
    if value < 1:
        raise typer.BadParameter("--moves takes a positive integer or 'unlimited'")
    return value


@app.command("generate")
def generate_command(
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("data"),
) -> None:
    """Write the synthetic layout, SKUs, 12 weeks of orders and the current slotting."""
    ds = generate(GenConfig(seed=seed))
    write_dataset(ds, out)
    o = ds.orders
    typer.echo(
        f"wrote {len(ds.layout)} slots, {ds.n_skus} SKUs, {o.n_orders} orders "
        f"({o.n_lines} lines) to {out}"
    )


@app.command()
def profile(
    data: DataOpt = Path("data"),
    out: OutOpt = Path("out"),
    first_week: Annotated[int, typer.Option(help="First week of the window.")] = 1,
    last_week: Annotated[int, typer.Option(help="Last week (default: final week).")] = 0,
) -> None:
    """Velocity, cube movement, cube-per-order index, size eligibility and affinity pairs."""
    ds = _load(data)
    last = last_week or ds.orders.n_weeks
    prof = profile_skus(ds, first_week, last)
    out.mkdir(parents=True, exist_ok=True)
    write_profile(ds, prof, out / "profile.csv")

    lines = np.sort(prof.lines_per_week)[::-1]
    total = lines.sum()
    typer.echo(f"weeks {first_week}-{last}: {total:.0f} pick lines per week over {ds.n_skus} SKUs")
    for share in (0.1, 0.2):
        k = int(round(share * ds.n_skus))
        typer.echo(
            f"top {int(share * 100)}% of SKUs ({k}) carry {100 * lines[:k].sum() / total:.1f}%"
        )
    need = np.bincount(ds.need_rank, minlength=3)
    typer.echo(
        f"size need: S {need[0]}, M {need[1]}, L {need[2]}; slots: {ds.layout.count_by_size()}"
    )
    pairs = mine_affinity(ds.orders, first_week, last)
    typer.echo(f"affinity pairs (support >= 0.5%, lift >= 3): {len(pairs)}")
    for p in pairs[:5]:
        typer.echo(
            f"  {ds.skus[p.a].sku} + {ds.skus[p.b].sku}  support {100 * p.support:.2f}%  "
            f"lift {p.lift:.1f}"
        )
    typer.echo(f"wrote {out / 'profile.csv'}")


@app.command()
def plan(
    objective: Annotated[str, typer.Option(help="aisle or depot.")] = "aisle",
    moves: Annotated[str, typer.Option(help="Move budget: an integer or 'unlimited'.")] = "150",
    data: DataOpt = Path("data"),
    out: OutOpt = Path("out"),
    train_weeks: TrainOpt = TRAIN_WEEKS,
) -> None:
    """Fit a target slotting on the training weeks and write the ranked moves to moves.csv."""
    if objective not in OBJECTIVES:
        raise typer.BadParameter(f"--objective must be one of {', '.join(OBJECTIVES)}")
    budget = _parse_budget(moves)
    ds = _load(data)
    target = fit_target(ds, objective, train_weeks)
    plan_moves = budgeted_moves(ds, target, budget)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "moves.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sku", "from", "to", "saving"])
        for m in plan_moves:
            slots = ds.layout.slots
            writer.writerow(
                [ds.skus[m.sku].sku, slots[m.from_slot].slot_id, slots[m.to_slot].slot_id,
                 f"{m.saving:.3f}"]
            )  # fmt: skip
    with (out / f"target_{objective}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sku", "slot"])
        for s, slot in zip(ds.skus, target.slot_of, strict=True):
            writer.writerow([s.sku, ds.layout.slots[int(slot)].slot_id])

    cost = target.slot_cost
    now = float((target.picks * cost[ds.current]).sum())
    full = float((target.picks * cost[target.slot_of]).sum())
    swaps = len(target.affinity.accepted) if target.affinity else 0
    typer.echo(f"objective {objective}, fitted on weeks 1-{train_weeks}")
    typer.echo(f"proxy cost per week: current {now:.0f}, full re-slot {full:.0f}")
    typer.echo(f"affinity swaps accepted: {swaps}")
    typer.echo(
        f"{len(plan_moves)} moves planned, proxy saving {sum(m.saving for m in plan_moves):.0f} "
        f"of {now - full:.0f} pick-metres/week"
    )
    typer.echo(f"wrote {out / 'moves.csv'} and {out / f'target_{objective}.csv'}")


@app.command()
def replay(
    data: DataOpt = Path("data"),
    out: OutOpt = Path("out"),
    train_weeks: TrainOpt = TRAIN_WEEKS,
) -> None:
    """Score current, full re-slot and budgeted plans on training and held-out weeks."""
    ds = _load(data)
    result = run_replay(ds, train_weeks)
    typer.echo(replay_table(result))
    typer.echo("")
    typer.echo(optimism_sentence(result))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replay.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["plan", "moves", "train_m_per_order", "held_out_m_per_order",
                         "held_out_largest_gap_m_per_order"])  # fmt: skip
        for r in result.rows:
            writer.writerow([r.name, r.moves, f"{r.train:.4f}", f"{r.held_out:.4f}",
                             f"{r.held_out_largest_gap:.4f}"])  # fmt: skip
    typer.echo(f"wrote {out / 'replay.csv'}")


@app.command()
def report(
    data: DataOpt = Path("data"),
    out: OutOpt = Path("out"),
    train_weeks: TrainOpt = TRAIN_WEEKS,
    moves: Annotated[
        str, typer.Option(help="Budget of the plan whose top moves are listed.")
    ] = "150",
) -> None:
    """Write report.md: replay table, top moves and affinity pairs."""
    budget = _parse_budget(moves)
    ds = _load(data)
    result = run_replay(ds, train_weeks, budgets=tuple(dict.fromkeys((50, 150, 300, None, budget))))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(ds, result, budget))
    typer.echo(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    app()
