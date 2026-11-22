import matplotlib

matplotlib.use("Agg")  # no screen needed when tests run on GitHub

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def monthly_returns(rng):
    """120 months x 9 sectors of random returns."""
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    cols = [f"S{i}" for i in range(9)]
    return pd.DataFrame(rng.normal(0.008, 0.05, (120, 9)), index=idx, columns=cols)
