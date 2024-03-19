from rtanalysis.generate_testdata import generate_test_df
from rtanalysis.rtanalysis import RTAnalysis
import pytest
import pandas as pd


def converter(series, convert_to):
    if convert_to == "list":
        return series.to_list()
    else:
        return series.to_numpy()


@pytest.mark.parametrize("convert_to", ["list", "nparray"])
@pytest.mark.parametrize("var", ["rt", "accuracy"])
def test_type_series(convert_to, var):
    """Whether giving as rt or accuracy inputs as lists or numpy arrays converts them to pd.Series"""
    test_df = generate_test_df(2, 1, 0.8)
    # convert var to list or numpy array
    var = converter(test_df[var], convert_to)

    rta = RTAnalysis()

    assert isinstance(rta._ensure_series_type(var), pd.Series)
