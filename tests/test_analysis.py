import pytest
import numpy as np

from oneqmc.analysis.h5_io import ffill0


@pytest.mark.parametrize(
    "x,expected",
    [
        [np.array([1.0, np.nan, 2])[:, None], np.array([1.0, 1, 2])[:, None]],
        [
            np.array([[np.nan, np.nan], [0.0, 2.0], [1.0, np.nan]]),
            np.array([[np.nan, np.nan], [0.0, 2.0], [1.0, 2.0]]),
        ],
    ],
)
def test_ffill0(x, expected):
    filled = ffill0(x)
    np.testing.assert_allclose(filled, expected)
