"""Plot relaxation of MTE problem to convex problem with larger parameter space."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from lp_relax.config import BLD

# Build matrices from scratch, then optimize.
# For the non-linear one, add the unit-ball constraint

# TODO(@buddeul): fix this after re-running: Currently force file collection and not
# depending on computation task. This was to avoid long runtimes.

path_to_results = BLD / "data" / "relaxation"

algorithms_to_plot = ["scipy_slsqp", "scipy_cobyla"]


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    path_to_plot_html: Annotated[Path, Product]
    k_bernstein: int
    algorithm: str


KWARGS = {
    f"k_bernstein_11_{algorithm}": _Arguments(
        path_to_plot=BLD
        / "figures"
        / "relaxation"
        / f"relaxation_mte_k_bernstein_11_{algorithm}.png",
        path_to_plot_html=BLD
        / "figures"
        / "relaxation"
        / f"relaxation_mte_k_bernstein_11_{algorithm}.html",
        k_bernstein=11,
        algorithm=algorithm,
    )
    for algorithm in algorithms_to_plot
}

for id_, kwargs in KWARGS.items():

    @pytask.mark.relax_mte_plots
    @task(name=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_relaxation_mte(
        path_to_plot: Annotated[Path, Product],
        path_to_plot_html: Annotated[Path, Product],
        k_bernstein: int,
        algorithm: str,
    ) -> None:
        """Task for solving original and relaxed convex problem."""
        # Load results from identification task
        results = []

        # Get all pkl files in path_to_results
        files_results = list(Path(path_to_results).rglob("*.pkl"))

        for path in files_results:
            with Path.open(path, "rb") as file:
                results.append(pickle.load(file))

        # Merge the dataframes
        data = pd.concat(results)

        data = data[data["k_bernstein"] == k_bernstein]
        data = data[data["algorithm"] == algorithm]

        # Plot the results by k_approximation

        fig = go.Figure()

        for k_approximation in data["k_approximation"].unique():
            data_k = data[data["k_approximation"] == k_approximation].reset_index(
                names="beta",
            )

            if k_approximation == data["k_approximation"].unique()[0]:
                fig.add_trace(
                    go.Scatter(
                        x=data_k["beta"],
                        y=data_k["lp"],
                        mode="lines",
                        name="LP Solution",
                        legendgroup="lp",
                    ),
                )

            fig.add_trace(
                go.Scatter(
                    x=data_k["beta"],
                    y=data_k["convex"],
                    mode="lines",
                    name=f"Convex, k = {k_approximation}",
                    legendgroup=f"k = {k_approximation}",
                ),
            )

        fig.update_layout(
            title="Relaxation of MTE Problem to Convex Problem",
            xaxis_title="Beta",
            yaxis_title="Value",
            legend_title="",
        )

        fig.write_image(path_to_plot)
        fig.write_html(path_to_plot_html)


@pytask.mark.relax_mte_plots
def task_plot_relaxation_mte_multiple_algorithms(
    k_approximation: int = 20,
    k_bernstein: int = 11,
    path_to_plot: Annotated[Path, Product] = (
        BLD
        / "figures"
        / "relaxation"
        / "relaxation_mte_k_bernstein_11_k_20_by_algorithm.png"
    ),
) -> None:
    """Task for solving original and relaxed convex problem."""
    # Load results from identification task
    results = []

    # Get all pkl files in path_to_results
    files_results = list(Path(path_to_results).rglob("*.pkl"))

    for path in files_results:
        with Path.open(path, "rb") as file:
            results.append(pickle.load(file))

    # Merge the dataframes
    data = pd.concat(results)

    data = data[data["k_bernstein"] == k_bernstein]
    data = data[data["k_approximation"] == k_approximation]

    # Plot the results by k_approximation

    fig = go.Figure()

    for algorithm in algorithms_to_plot:
        data_algorithm = data[data["algorithm"] == algorithm].reset_index(
            names="beta",
        )

        fig.add_trace(
            go.Scatter(
                x=data_algorithm["beta"],
                y=data_algorithm["convex"],
                mode="lines",
                name=f"{algorithm}",
                legendgroup=f"{algorithm}",
            ),
        )

    # Plot the LP solution
    data_lp = data[data["algorithm"] == "scipy_slsqp"].reset_index(names="beta")

    fig.add_trace(
        go.Scatter(
            x=data_lp["beta"],
            y=data_lp["lp"],
            mode="lines",
            name="LP Solution",
            legendgroup="lp",
        ),
    )

    subtitle = (
        f"<br><sup> Bernstein Degree {k_bernstein},"
        f"K = {k_approximation}, No shape restrictions</sup>"
    )

    fig.update_layout(
        title=(
            "Relaxation of MTE Problem to Convex Problem: Algorithm Comparison"
            + subtitle
        ),
        xaxis_title="Beta",
        yaxis_title="Value",
        legend_title="",
    )

    fig.write_image(path_to_plot)
