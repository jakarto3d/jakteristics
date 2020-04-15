from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import List

import typer
from typer import Option, Argument

from . import extension, las_utils
from .constants import FEATURE_NAMES


app = typer.Typer()


def show_features_callback(value):
    if value:
        for name in FEATURE_NAMES:
            typer.echo(name)
        raise typer.Exit()


@app.command()
def typer_main(
    las_input: Path = Argument(...),
    output: Path = Argument(...),
    search_radius: float = Option(
        ...,
        "--search-radius",
        "-s",
        help="The search radius to use to query neighbors.",
    ),
    num_threads: int = Option(
        -1,
        "--num_threads",
        "-t",
        help=(
            "The number of threads to use for computation. "
            "Defaults to the number of cpu on the machine."
        ),
    ),
    feature_names: List[str] = Option(
        FEATURE_NAMES,
        "--feature",
        "-f",
        help=(
            "The feature names to compute. Repeat this parameter "
            "to compute multiple features. "
            "Use --show-features to see the list of possible choices. "
            "Default: All features."
        ),
    ),
    manhattan_distance: bool = Option(
        False,
        "--manhattan-distance",
        "-m",
        is_flag=True,
        help=(
            "How to compute the distance between 2 points. "
            "If provided, the sum-of-absolute-values is used ('Manhattan' distance)."
            "By default, the standard Euclidean distance is used. "
        ),
    ),
    eps: float = Option(
        0,
        "--eps",
        "-e",
        show_default=True,
        help=(
            "Return approximate nearest neighbors; the k-th returned value "
            "is guaranteed to be no further than (1+eps) times the "
            "distance to the real k-th nearest neighbor."
        ),
    ),
    show_features: bool = typer.Option(
        None,
        "--show-features",
        help="Show a list of possible feature names and exit.",
        callback=show_features_callback,
        is_eager=True,
    ),
):
    t = perf_counter()
    typer.echo("Computing geometric features...")

    xyz = las_utils.read_las_xyz(las_input)
    feature_names_str = []
    for name in feature_names:
        if name not in FEATURE_NAMES:
            choices = ", ".join(FEATURE_NAMES)
            raise typer.BadParameter(f"invalid choice: {name}. (choose from {choices})")
        feature_names_str.append(name)

    features = extension.compute_features(
        xyz,
        search_radius,
        num_threads=num_threads,
        euclidean_distance=not manhattan_distance,
        eps=eps,
        feature_names=feature_names_str,
    )

    las_utils.write_with_extra_dims(las_input, output, features, feature_names_str)
    typer.echo(f"Done in {perf_counter() - t:0.2f} seconds.")


def main():
    app()


# used for documentation
click_command = typer.main.get_command(app)

if __name__ == "__main__":
    app()
