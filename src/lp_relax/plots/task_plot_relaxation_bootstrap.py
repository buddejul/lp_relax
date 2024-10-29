"""Tasks to plot results from relaxation bootstrap."""

import shutil
import tarfile
from pathlib import Path
from typing import Annotated

import pandas as pd  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product

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


@pytask.mark.local
def task_plot_coverage_by_method(
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

    data = combined.groupby(["method", "slope"]).mean().reset_index()

    fig = go.Figure()

    for method in data.method.unique():
        sub_data = data[data.method == method]

        fig.add_trace(
            go.Scatter(
                y=sub_data["covers_lower_one_sided"],
                x=sub_data["slope"],
                mode="lines+markers",
                name=f"{method.replace('_', ' ').capitalize()}",
            ),
        )

    fig.update_layout(
        title="Coverage by Method and Parameter",
        xaxis_title="Slope",
        yaxis_title="Coverage",
    )

    # Add note: Data is Normal with sigma = 1
    fig.add_annotation(
        text="Data: Normal(, 1)",
        xref="paper",
        yref="paper",
        x=1,
        y=-0.1,
        showarrow=False,
    )

    fig.write_html(path_to_plot_html)
    fig.write_html(path_to_plot_png)
