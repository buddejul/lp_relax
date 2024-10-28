"""Task for bootstrap simulation for relaxation of example problem."""

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
import pytask
from pytask import Product, task

from lp_relax.config import BLD
from lp_relax.sims.funcs import simulation

num_points = 20
start_left = -0.2
end_right = 0.2

slopes = np.concatenate(
    (
        np.linspace(
            start_left,
            0,
            np.floor(num_points / 2).astype(int),
            endpoint=False,
        ),
        np.linspace(0, end_right, np.ceil(num_points / 2).astype(int)),
    ),
)


class _Arguments(NamedTuple):
    slope: float
    path_to_results: Annotated[Path, Product]
    num_sims: int = 20
    num_boot: int = 1000
    num_obs: int = 1000
    alpha: float = 0.05


ID_TO_KWARGS = {
    f"slope_{slope}": _Arguments(
        slope=slope,
        path_to_results=(
            BLD / "data" / "relaxation_bootstrap" / f"results_{slope}.pkl"
        ),
    )
    for slope in np.zeros(1)
}

for kwargs in ID_TO_KWARGS.values():

    @pytask.mark.hpc_relax_boot
    @task(kwargs=kwargs)  # type: ignore[arg-type]
    def task_relaxation_bootstrap(
        num_sims: int,
        num_boot: int,
        num_obs: int,
        slope: float,
        alpha: float,
        path_to_results: Annotated[Path, Product],
    ) -> None:
        """Task for bootstrap simulation for relaxation of problem."""
        results = simulation(
            num_sims=num_sims,
            num_boot=num_boot,
            num_obs=num_obs,
            slope=slope,
            alpha=alpha,
        )

        results["slope"] = slope

        results.reset_index(names="method").to_pickle(path_to_results)
