import csv

import pytest
from typer.testing import CliRunner

from reslot.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "data"
    out = runner.invoke(app, ["generate", "--seed", "7", "--out", str(path)])
    assert out.exit_code == 0, out.output
    return path


def test_generate_writes_every_input_file(data_dir):
    names = {p.name for p in data_dir.iterdir()}
    assert names == {"layout.csv", "skus.csv", "orders.csv", "slotting_current.csv", "truth.json"}


def test_profile_reports_the_recovered_pairs(data_dir, tmp_path):
    out = runner.invoke(app, ["profile", "--data", str(data_dir), "--out", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "affinity pairs (support >= 0.5%, lift >= 3): 25" in out.output
    assert (tmp_path / "profile.csv").exists()


def test_plan_writes_ranked_moves_csv_within_budget(data_dir, tmp_path):
    out = runner.invoke(
        app,
        ["plan", "--objective", "aisle", "--moves", "150", "--data", str(data_dir),
         "--out", str(tmp_path)],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    with (tmp_path / "moves.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["sku", "from", "to", "saving"]
    assert 100 < len(rows) <= 150
    assert float(rows[0]["saving"]) > 1000
    assert (tmp_path / "target_aisle.csv").exists()


def test_plan_is_deterministic_across_independent_runs(data_dir, tmp_path):
    other_data = tmp_path / "again"
    assert runner.invoke(app, ["generate", "--seed", "7", "--out", str(other_data)]).exit_code == 0
    outputs = []
    for data, name in ((data_dir, "a"), (other_data, "b")):
        out_dir = tmp_path / name
        args = ["plan", "--moves", "300", "--data", str(data), "--out", str(out_dir)]
        assert runner.invoke(app, args).exit_code == 0
        outputs.append((out_dir / "moves.csv").read_bytes())
    assert outputs[0] == outputs[1]
    assert (data_dir / "orders.csv").read_bytes() == (other_data / "orders.csv").read_bytes()


def test_plan_accepts_unlimited_and_the_depot_objective(data_dir, tmp_path):
    base = ["--data", str(data_dir), "--out", str(tmp_path)]
    ok = runner.invoke(app, ["plan", "--objective", "depot", "--moves", "unlimited", *base])
    assert ok.exit_code == 0, ok.output
    assert "affinity swaps accepted" in ok.output


@pytest.mark.parametrize("args", [["--objective", "cube"], ["--moves", "0"], ["--moves", "lots"]])
def test_plan_rejects_bad_options(data_dir, tmp_path, args):
    out = runner.invoke(app, ["plan", *args, "--data", str(data_dir), "--out", str(tmp_path)])
    assert out.exit_code != 0


def test_commands_fail_cleanly_without_data(tmp_path):
    out = runner.invoke(app, ["replay", "--data", str(tmp_path / "missing")])
    assert out.exit_code != 0


def test_replay_and_report_commands(data_dir, tmp_path):
    base = ["--data", str(data_dir), "--out", str(tmp_path)]
    rep = runner.invoke(app, ["replay", *base])
    assert rep.exit_code == 0, rep.output
    assert "held-out vs current" in rep.output and "optimism gap" in rep.output
    assert (tmp_path / "replay.csv").exists()
    rpt = runner.invoke(app, ["report", *base])
    assert rpt.exit_code == 0, rpt.output
    text = (tmp_path / "report.md").read_text()
    assert "## Top moves" in text and "full re-slot, aisle" in text
