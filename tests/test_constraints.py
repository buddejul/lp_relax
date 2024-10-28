"""Tests constraints are working."""

import numpy as np
import optimagic as om  # type: ignore[import-untyped]

from lp_relax.funcs.lp_relax import (
    generate_poly_constraints,
    generate_poly_constraints_new,
    generate_sphere_constraint,
    solve_lp_convex,
)


def test_generate_poly_constraints():
    num_dims = 2
    s = 100
    k = 4

    constraints_to_test = [
        generate_poly_constraints_new,
        # generate_poly_constraints,
        # generate_sphere_constraint,
    ]

    for generate_constraints in constraints_to_test:
        if generate_constraints == generate_poly_constraints_new:
            kwargs = {"num_dims": num_dims, "s": s}
        elif generate_constraints == generate_poly_constraints:
            kwargs = {"num_dims": num_dims}
        elif generate_constraints == generate_sphere_constraint:
            kwargs = {"num_dims": num_dims, "k": k}

        constraints = generate_constraints(**kwargs)

        # Check fthat (0, 0), (1, 1), (0, 1), (1, 0) are in the constraints
        for x in [(0, 0), (1, 1), (0, 1), (1, 0)]:
            assert np.all(
                [
                    constraint.func(x) <= constraint.upper_bound
                    for constraint in constraints
                ]
            )


def test_om_with_poly_constraint():
    num_dims = 2

    constraints = generate_poly_constraints(num_dims)

    om.minimize(
        fun=lambda x: np.sum(x**2),
        constraints=constraints,
        params=np.ones(num_dims),
        algorithm="ipopt",
    )


def test_solve_lp_convex():
    solve_lp_convex(
        beta=0.5, algorithm="ipopt", constraint_type="poly_old", s=10, k_bernstein=11
    )
