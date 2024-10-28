"""Utilities used throughout the project."""

import numpy as np


def make_mtr_binary_iv(
    yd_c: float | np.ndarray,
    yd_at: float | np.ndarray,
    yd_nt: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
):
    """Construct MTR with constant splines for binary IV."""
    _pscores = {
        "pscore_lo": pscore_lo,
        "pscore_hi": pscore_hi,
    }

    def mtr(u):
        return (
            yd_at * _at(u, **_pscores)
            + yd_c * _c(u, **_pscores)
            + yd_nt * _nt(u, **_pscores)
        )

    return mtr


def _at(
    u: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
) -> bool | np.ndarray:
    del pscore_hi
    return u <= pscore_lo


def _c(
    u: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
) -> bool | np.ndarray:
    return pscore_lo <= u and pscore_hi > u


def _nt(
    u: float | np.ndarray,
    pscore_lo: float | np.ndarray,
    pscore_hi: float | np.ndarray,
) -> bool | np.ndarray:
    del pscore_lo
    return u >= pscore_hi
