from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis
import pytest
import numpy as np


@pytest.mark.parametrize("outlier_cutoff_sd", np.linspace(0.90, 0.99, 5))
def test_outlier_removal(outlier_cutoff_sd):
    """Whether reaction times only consist of values less than the given cutoff"""
    test_df = generate_test_df(2, 1, 0.8)

    rta = RTAnalysis(outlier_cutoff_sd)

    rt = rta.reject_outlier_rt(test_df.rt)
    rt_masked = rt.dropna().to_numpy()
    cutoff = test_df.rt.std() * outlier_cutoff_sd

    # we are testing pretty high cutoffs, so it should not remove all RTs
    if rt_masked.shape[0] == 0:
        raise ValueError(
            f"outlier reject excluded all trials at {outlier_cutoff_sd} cutoff "
        )
    else:
        assert (rt_masked <= cutoff).all()
