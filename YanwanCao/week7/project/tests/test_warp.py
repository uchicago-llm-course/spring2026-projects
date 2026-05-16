import numpy as np
import pytest
from src.hybrid_bo.data.warp import warp, unwarp, config_to_warped, warped_to_config

@pytest.mark.parametrize("transform,lo,hi,val", [
    ("log",    1e-2, 1e5,  10.0),
    ("log",    1e-6, 1e0,  1e-3),
    ("logit",  0.05, 0.95, 0.5),
    ("linear", 1.0,  32.0, 16.0),
])
def test_warp_roundtrip(transform, lo, hi, val):
    """warp then unwarp should recover the original value."""
    warped   = warp(val, transform, lo, hi)
    recovered = unwarp(warped, transform, lo, hi)
    assert abs(recovered - val) < 1e-9, f"roundtrip failed: {val} -> {warped} -> {recovered}"

def test_warp_output_in_unit_interval():
    for transform, lo, hi, val in [("log", 1e-2, 1e5, 100.0), ("linear", 0, 1, 0.5)]:
        w = warp(val, transform, lo, hi)
        assert 0.0 <= w <= 1.0

def test_config_to_warped_shape(svm_space):
    config = {"C": 1.0, "gamma": 1e-3, "tol": 1e-4}
    arr = config_to_warped(config, svm_space)
    assert arr.shape == (3,)
    assert np.all((arr >= 0) & (arr <= 1))