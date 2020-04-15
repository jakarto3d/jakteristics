import typer
from typer.testing import CliRunner

runner = CliRunner()

from jakteristics.__main__ import typer_main


app = typer.Typer()
app.command()(typer_main)


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
