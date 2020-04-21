from jakteristics.constants import FEATURE_NAMES
from pathlib import Path
import shutil

import pytest
import typer
from typer.testing import CliRunner

from jakteristics.__main__ import typer_main


runner = CliRunner()
app = typer.Typer()
app.command()(typer_main)

TEST_DATA = Path(__file__).parent / "data"


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_show_features():
    result = runner.invoke(app, ["--show-features"])
    assert result.stdout == "\n".join(FEATURE_NAMES) + "\n"
    assert result.exit_code == 0


def test_cli_invalid_feature_name(temp_dir):
    input_file = TEST_DATA / "test_0.02_seconde.las"
    output_file = temp_dir / "out.las"
    result = runner.invoke(
        app,
        [
            str(input_file),
            str(output_file),
            "--search-radius",
            "0.15",
            "--feature",
            "bananas",
        ],
    )
    assert result.exit_code == 2
    assert "invalid choice: bananas" in result.stdout


def test_cli_run_basic(temp_dir):
    input_file = TEST_DATA / "test_0.02_seconde.las"
    output_file = temp_dir / "non_existing_folder" / "out.las"
    result = runner.invoke(
        app,
        [
            str(input_file),
            str(output_file),
            "--search-radius",
            "0.15",
            "--num-threads",
            "4",
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists
