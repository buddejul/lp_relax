"""Plot task for figures in paper."""

from functools import partial
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from lp_relax.config import BLD
from lp_relax.sims.funcs import problems_to_sim

num_points = 1000

slope_grid = np.linspace(-1, 1, num_points)


class _Arguments(NamedTuple):
    path_to_plot: Annotated[Path, Product]
    slope_grid: np.ndarray = slope_grid
    problems: dict[str, partial] = problems_to_sim


ID_TO_KWARGS = {
    f"plot_{slope_grid}": _Arguments(
        path_to_plot=BLD.parent
        / "documents"
        / "figures"
        / "value_function_by_constraint_set_dim_2.png",
        slope_grid=slope_grid,
        problems=problems_to_sim,
    )
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.local
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_relaxation_paper(
        path_to_plot: Annotated[Path, Product],
        slope_grid: np.ndarray,
        problems: dict[str, partial],
    ) -> None:
        # Split problems into linear and convex
        convex_problems = {k: v for k, v in problems.items() if k != "linear"}

        funs = {k: np.zeros_like(slope_grid) for k in problems}
        solutions = {k: np.zeros((len(slope_grid), 2)) for k in problems}

        for i, val in enumerate(slope_grid):
            _res_linear = problems["linear"](fun_kwargs={"slope": val})
            funs["linear"][i] = _res_linear.fun
            solutions["linear"][i] = _res_linear.params

            for k, problem in convex_problems.items():
                _res_convex = problem(fun_kwargs={"slope": val})
                funs[k][i] = _res_convex.fun
                solutions[k][i] = _res_convex.params

                fig = go.Figure()

        # Plot fun_linear and fun_convex against grid

        fig.add_trace(
            go.Scatter(
                x=slope_grid,
                y=funs["linear"],
                mode="lines",
                name="",
                legendgroup="linear",
                legendgrouptitle={"text": "Linear Constraint"},
            )
        )

        for k in convex_problems:
            fig.add_trace(
                go.Scatter(
                    x=slope_grid,
                    y=funs[k],
                    mode="lines",
                    legendgroup="convex",
                    name=f"k = {k.split('_')[-1]}",
                    legendgrouptitle={"text": "Convex Constraints"},
                )
            )

        # Add note with number of grid points
        fig.add_annotation(
            x=0.75,
            y=-1,
            text=f"Number of grid points: {num_points}",
            showarrow=False,
            align="center",
        )

        # Titles
        fig.update_layout(
            title="Value Function by Constraint Set and Slope Parameter",
            xaxis_title="Slope of x1 (c1)",
            yaxis_title="Minimized Objective",
        )

        fig.show()

        fig.write_image(
            path_to_plot,
            scale=2,
            width=800,
            height=600,
        )
