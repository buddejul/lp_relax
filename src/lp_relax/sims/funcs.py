"""Funcs for bootstrap simulations."""

from functools import partial

import numpy as np
import optimagic as om  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from lp_relax.config import RNG
from lp_relax.funcs.lp_relax import (
    generate_poly_constraints,
)


def _linear_objective(params, slope):
    return -(slope * params[0] + params[1])


bounds = om.Bounds(lower=np.array([0, 0]), upper=np.array([1, 1]))

linear = partial(
    om.minimize,
    fun=_linear_objective,
    params=np.ones(2) * 0.5,
    algorithm="scipy_lbfgsb",
    bounds=bounds,
)


constraint = om.NonlinearConstraint(
    func=lambda x: np.sum((x - 0.5) ** 2),
    upper_bound=0.5,
)

convex = partial(
    om.minimize,
    fun=_linear_objective,
    params=np.ones(2) * 0.5,
    algorithm="scipy_cobyla",
    constraints=[constraint],
)

constraints = generate_poly_constraints(2)

convex_sharper = partial(
    om.minimize,
    fun=_linear_objective,
    params=np.ones(2) * 0.5,
    algorithm="scipy_cobyla",
    constraints=constraints,
)


# Now do a bootstrap simulation
def _draw_data(num_obs: int, mean: float, sigma: float = 1) -> np.ndarray:
    return RNG.normal(loc=mean, scale=sigma, size=num_obs)


def _bootstrap_relax(
    num_boot: int,
    num_obs: int,
    slope: float,
    alpha: float,
) -> pd.DataFrame:
    """Bootstrap simulation for relaxation of example problem."""
    problems = [linear, convex, convex_sharper]

    # ----------------------------------------------------------------------------------
    # Generate Data
    # ----------------------------------------------------------------------------------
    data = _draw_data(num_obs=num_obs, mean=slope)

    data_fun = np.zeros(len(problems))

    data_slope = np.mean(data)

    for i, problem in enumerate(problems):
        _res = problem(fun_kwargs={"slope": data_slope})
        data_fun[i] = _res.fun

    # ----------------------------------------------------------------------------------
    # Bootstrap
    # ----------------------------------------------------------------------------------
    boot_slope = np.zeros(num_boot)
    boot_fun = np.zeros((num_boot, len(problems)))

    for i in range(num_boot):
        data_boot = RNG.choice(data, size=num_obs, replace=True)

        boot_slope[i] = np.mean(data_boot)

        for j, problem in enumerate(problems):
            _res = problem(fun_kwargs={"slope": boot_slope[i]})
            boot_fun[i, j] = _res.fun

    # ----------------------------------------------------------------------------------
    # Confidence Intervals
    # ----------------------------------------------------------------------------------
    rn = np.sqrt(num_obs)

    # Center the bootstrap distribution: rn * (boot_fun - data_fun)
    boot_fun_centered = rn * (boot_fun - data_fun)

    # For lower CIs: Take 1 - alpha/2 quantile
    crit_lower_ci = np.quantile(boot_fun_centered, 1 - alpha / 2, axis=0)
    crit_lower_ci_one_sided = np.quantile(boot_fun_centered, 1 - alpha, axis=0)

    # For upper CIs: Take alpha/2 quantile
    crit_upper_ci = np.quantile(boot_fun_centered, alpha / 2, axis=0)
    crit_upper_ci_one_sided = np.quantile(boot_fun_centered, alpha, axis=0)

    # Construct CIs
    lower_ci = data_fun - crit_lower_ci / rn
    upper_ci = data_fun - crit_upper_ci / rn

    lower_ci_one_sided = data_fun - crit_lower_ci_one_sided / rn
    upper_ci_one_sided = data_fun - crit_upper_ci_one_sided / rn

    out = pd.DataFrame(
        {
            "data_fun": data_fun,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "lower_ci_one_sided": lower_ci_one_sided,
            "upper_ci_one_sided": upper_ci_one_sided,
        },
        index=["linear", "convex", "convex_sharper"],
    )

    out["true_fun_linear"] = linear(fun_kwargs={"slope": slope}).fun

    return out


def simulation(
    num_sims: int,
    num_boot: int,
    num_obs: int,
    slope: float,
    alpha: float,
) -> pd.DataFrame:
    """Perform bootstrap simulation for relaxation of example problem."""
    results = [
        _bootstrap_relax(num_boot, num_obs, slope, alpha) for i in range(num_sims)
    ]

    data = pd.concat(results, axis=0)

    data["covers"] = (data["lower_ci"] <= data["true_fun_linear"]) & (
        data["upper_ci"] >= data["true_fun_linear"]
    )

    data["covers_lower_one_sided"] = (
        data["lower_ci_one_sided"] <= data["true_fun_linear"]
    )

    return data
