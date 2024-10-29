"""Tasks to plot results from relaxation bootstrap."""

import shutil
import tarfile
from pathlib import Path
from typing import Annotated, NamedTuple

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from lp_relax.config import BLD, SRC

JOBIDS_TO_PLOT = [17692621]

RES_FILES_TAR = [SRC / "marvin" / f"{jobid}.tar.gz" for jobid in JOBIDS_TO_PLOT]


@pytask.mark.local
def task_combine_relaxation_bootstrap_results(
    res_files_tar: list[Path] = RES_FILES_TAR,
    path_to_combined: Annotated[Path, Product] = (
        BLD / "data" / "relaxation_bootstrap" / "relaxation_bootstrap_combined.pkl"
    ),
) -> None:
    """Combine results from relaxation bootstrap tasks."""
    # Unzip files in res_files into a temporary directory
    tmp_dir = BLD / "marvin" / "_tmp"

    for file in res_files_tar:
        with tarfile.open(file, "r:gz") as tar:
            tar.extractall(path=tmp_dir, filter="data")

    # Collect al lfile names in the temporary directory and all its subdirectories
    res_files = list(tmp_dir.rglob("*.pkl"))

    dfs = [pd.read_pickle(file) for file in res_files]

    out = pd.concat(dfs, ignore_index=True)

    out.to_pickle(path_to_combined)

    # Remove temporary directory
    shutil.rmtree(tmp_dir)


class _Arguments(NamedTuple):
    path_to_plot_html: Annotated[Path, Product]
    path_to_plot_png: Annotated[Path, Product]
    stat_to_plot: str


ID_TO_KWARGS = {
    f"{stat_to_plot}": _Arguments(
        path_to_plot_html=BLD
        / "figures"
        / "relaxation_bootstrap"
        / f"{stat_to_plot}_by_method.html",
        path_to_plot_png=BLD
        / "figures"
        / "relaxation_bootstrap"
        / f"{stat_to_plot}_by_method.png",
        stat_to_plot=stat_to_plot,
    )
    for stat_to_plot in ["covers_lower_one_sided", "lower_ci_one_sided"]
}

for id_, kwargs in ID_TO_KWARGS.items():

    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    @pytask.mark.local
    def task_plot_coverage_by_method(
        stat_to_plot: str,
        path_to_combined: Path = BLD
        / "data"
        / "relaxation_bootstrap"
        / "relaxation_bootstrap_combined.pkl",
        path_to_plot_html: Annotated[Path, Product] = BLD
        / "figures"
        / "relaxation_bootstrap"
        / "coverage_by_method.html",
        path_to_plot_png: Annotated[Path, Product] = BLD
        / "figures"
        / "relaxation_bootstrap"
        / "coverage_by_method.png",
    ) -> None:
        """Plot coverage by method and slope parameter."""
        combined = pd.read_pickle(path_to_combined)

        color_to_k_convex = {
            "linear": "blue",
            "convex_sphere_2": "red",
            "convex_sphere_4": "green",
            "convex_sphere_10": "purple",
        }

        data = combined.groupby(["method", "slope", "num_obs"]).mean().reset_index()

        num_obs_to_plot = data["num_obs"].unique()
        methods_to_plot = data.method.unique()

        fig = go.Figure()

        num_obs_to_dash = {1_000: "solid", 10_000: "dash"}

        for num_obs in num_obs_to_plot:
            for method in methods_to_plot:
                sub_data = data[data.method == method]
                sub_data = sub_data[sub_data.num_obs == num_obs]

                fig.add_trace(
                    go.Scatter(
                        y=sub_data[stat_to_plot],
                        x=sub_data["slope"],
                        mode="lines+markers",
                        line=dict(
                            dash=num_obs_to_dash[num_obs],
                            color=color_to_k_convex[method],
                        ),
                        marker=dict(color=color_to_k_convex[method]),
                        name=f"N={num_obs}",
                        legendgroup=method,
                        legendgrouptitle=dict(
                            text=f"{method.replace('_', ' ').capitalize()}"
                        ),
                    ),
                )

        stat_to_title = {
            "covers_lower_one_sided": "Coverage Lower One-Sided CI",
            "lower_ci_one_sided": "Mean Lower One-Sided CI",
        }

        subtitle = "<br><sup>Nominal Coverage 95%</sup>"

        stat_to_x_axis_title = {
            "covers_lower_one_sided": "Coverage",
            "lower_ci_one_sided": "Average",
        }

        fig.update_layout(
            title=(
                f"{stat_to_title[stat_to_plot]} by Method and Parameter" f"{subtitle}"
            ),
            xaxis_title="Slope",
            yaxis_title=f"{stat_to_x_axis_title[stat_to_plot]}",
        )

        # Add note: Data is Normal with sigma = 1
        fig.add_annotation(
            text="Data: Normal(Slope, 1)",
            xref="paper",
            yref="paper",
            x=1,
            y=-0.2,
            showarrow=False,
        )

        fig.write_html(path_to_plot_html)
        fig.write_image(path_to_plot_png)
