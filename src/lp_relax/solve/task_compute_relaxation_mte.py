"""Relaxation of MTE problem to convex problem with larger parameter space."""

import pickle
from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytask
from pytask import Product, task

from lp_relax.config import BLD
from lp_relax.funcs.lp_relax import solve_lp_convex

algorithms_to_compute = ["scipy_slsqp", "scipy_cobyla"]


class _Arguments(NamedTuple):
    path_to_data: Annotated[Path, Product]
    num_points: int
    k_bernstein: int
    k_approximation: int
    algorithm: str


ID_TO_KWARGS = {
    (
        f"k_bernstein_{k_bernstein}_" f"k_approximation_{k_approximation}_{algorithm}"
    ): _Arguments(
        path_to_data=BLD
        / "data"
        / "relaxation"
        / f"{algorithm}"
        / (
            f"relaxation_mte_k_bernstein_{k_bernstein}_"
            f"k_approximation_{k_approximation}.pkl"
        ),
        num_points=2000,
        k_bernstein=k_bernstein,
        k_approximation=k_approximation,
        algorithm=algorithm,
    )
    for k_bernstein in [11]
    for k_approximation in [2, 4, 10, 20, 40]
    for algorithm in algorithms_to_compute
    if not (k_bernstein == 5 and algorithm == "scipy_cobyla")  # noqa: PLR2004
}


for id_, kwargs in ID_TO_KWARGS.items():

    @pytask.mark.skip()
    @pytask.mark.relax
    @task(name=id_, kwargs=kwargs)  # type: ignore[arg-type]
    def task_relaxation_mte(
        path_to_data: Annotated[Path, Product],
        num_points: int,
        k_bernstein: int,
        k_approximation: int,
        algorithm: str,
    ) -> None:
        """Task for solving original and relaxed convex problem."""
        beta_grid = np.linspace(0.5, 1, num_points)

        res = [
            solve_lp_convex(
                beta=beta,
                k_bernstein=k_bernstein,
                k_approximation=k_approximation,
                algorithm=algorithm,
                constraint_type="sphere",
            )
            for beta in beta_grid
        ]

        # Put into DataFrame
        data = pd.DataFrame(res, index=beta_grid)
        data["num_points"] = num_points
        data["k_bernstein"] = k_bernstein
        data["k_approximation"] = k_approximation
        data["algorithm"] = algorithm

        # Save results
        with Path.open(path_to_data, "wb") as file:
            pickle.dump(data, file)
