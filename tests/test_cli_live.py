"""End-to-end smoke test for the live CLI command's state-path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from midas.cli import cli


def test_live_command_uses_sidecar_state_path_by_default(tmp_path: Path) -> None:
    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 10\n    cost_basis: 150\navailable_cash: 1000\n"
    )

    captured: dict[str, Path] = {}

    class _StubEngine:
        def __init__(self, *args: object, state_path: Path, **kwargs: object) -> None:
            captured["state_path"] = state_path

        def __enter__(self) -> _StubEngine:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        result = runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured["state_path"] == portfolio_yaml.with_suffix(".state.yaml")


def test_live_command_honors_state_file_field(tmp_path: Path) -> None:
    state_target = tmp_path / "run" / "explicit.state.yaml"
    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        f"state_file: {state_target}\n"
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 10\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )

    captured: dict[str, Path] = {}

    class _StubEngine:
        def __init__(self, *args: object, state_path: Path, **kwargs: object) -> None:
            captured["state_path"] = state_path

        def __enter__(self) -> _StubEngine:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        result = runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured["state_path"] == state_target
