"""Plot task for solution figures."""

from functools import partial
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from lp_relax.config import BLD
from lp_relax.sims.funcs import problems_to_sim

num_points = 2000

slope_grid = np.linspace(-1, 1, num_points)


class _Arguments(NamedTuple):
    path_to_plot_value: Annotated[Path, Product]
    path_to_plot_solutions: Annotated[Path, Product]
    slope_grid: np.ndarray = slope_grid
    problems: dict[str, partial] = problems_to_sim


ID_TO_KWARGS = {
    f"plot_{slope_grid}": _Arguments(
        path_to_plot_solutions=BLD
        / "figures"
        / "relaxation_bootstrap"
        / "solutions_by_constraint_set_dim_2.png",
        path_to_plot_value=BLD
        / "figures"
        / "relaxation_bootstrap"
        / "value_function_by_constraint_set_dim_2.png",
        slope_grid=slope_grid,
        problems=problems_to_sim,
    )
}

for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.local
    @task(id=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_plot_relaxation_solutions(
        path_to_plot_value: Annotated[Path, Product],
        path_to_plot_solutions: Annotated[Path, Product],
        slope_grid: np.ndarray,
        problems: dict[str, partial],
    ) -> None:
        color_linear = "blue"

        color_to_k_convex = {
            2: "red",
            4: "green",
            10: "purple",
            20: "gray",
        }

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
                line=dict(color=color_linear),
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
                    line=dict(color=color_to_k_convex[int(k.split("_")[-1])]),
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

        fig.write_image(
            path_to_plot_value,
            scale=2,
            width=800,
            height=600,
        )

        # Plot solutions (x,y) against slope
        fig = go.Figure()

        col_to_name = {0: "x1", 1: "x2"}

        dash_to_name = {
            "x1": "solid",
            "x2": "dash",
        }

        for col, name in col_to_name.items():
            show_legend = bool(name == "x1")

            fig.add_trace(
                go.Scatter(
                    x=slope_grid,
                    y=solutions["linear"][:, col],
                    mode="lines",
                    line_dash=dash_to_name[name],
                    line=dict(color=color_linear),
                    legendgroup="linear",
                    name="",
                    legendgrouptitle={"text": "Linear Constraint"},
                    showlegend=show_legend,
                )
            )

            for k in convex_problems:
                k_num = k.split("_")[-1]

                fig.add_trace(
                    go.Scatter(
                        x=slope_grid,
                        y=solutions[k][:, col],
                        mode="lines",
                        line=dict(color=color_to_k_convex[int(k_num)]),
                        line_dash=dash_to_name[name],
                        name=f"k = {k_num}",
                        legendgroup=f"convex, {name}",
                        legendgrouptitle={"text": "Convex Constraints"},
                        showlegend=show_legend,
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
            title="Solutions by Constraint Set and Slope Parameter",
            xaxis_title="Slope of x1 (c1)",
            yaxis_title="Optimal Solution",
        )

        fig.update_yaxes(range=[-0.5, 1])

        fig.write_image(
            path_to_plot_solutions,
            scale=2,
            width=800,
            height=600,
        )
