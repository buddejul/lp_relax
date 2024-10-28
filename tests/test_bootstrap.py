from lp_relax.sims.funcs import _bootstrap_relax


def test_bootstrap_relax():
    _bootstrap_relax(num_boot=100, num_obs=1000, slope=0, alpha=0.05)
