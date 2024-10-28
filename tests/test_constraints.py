"""Tests constraints are working."""

import numpy as np

from lp_relax.funcs.lp_relax import (
    generate_poly_constraints,
    generate_poly_constraints_new,
    generate_sphere_constraint,
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
