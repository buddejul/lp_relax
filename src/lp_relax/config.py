"""All the general configuration of the project."""

from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()


DOCUMENTS = ROOT.joinpath("documents").resolve()

Y1_AT = 0.75
Y0_AT = 0.25
Y1_NT = 0.45
Y0_NT = 0.2

RNG = np.random.default_rng()
