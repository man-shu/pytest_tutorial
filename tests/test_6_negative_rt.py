from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis
import pytest


def test_negative_rt_raises_error():
    """Whether giving negative response times (RT) as input raises ValueError"""
    test_df = generate_test_df(2, 1, 0.8)
    # convert reaction times to negative
    test_df["rt"] = test_df["rt"].multiply(-1)

    rta = RTAnalysis()
    with pytest.raises(ValueError):
        rta.fit(test_df.rt, test_df.accuracy)
