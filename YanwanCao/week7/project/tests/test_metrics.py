import numpy as np
from src.hybrid_bo.core.metrics import dist_w2, dist_mmd, dist_pointwise, dist_energy

def test_identical_distributions_have_zero_distance():
    rng = np.random.RandomState(0)
    X = rng.rand(50, 3)
    assert dist_w2(X, X)        < 1e-6
    assert dist_mmd(X, X)       < 1e-6
    assert dist_energy(X, X)    < 1e-6

def test_pointwise_known_value():
    X = np.array([[0.0, 0.0, 0.0]])
    Y = np.array([[3.0, 4.0, 0.0]])
    assert abs(dist_pointwise(X, Y) - 5.0) < 1e-9

def test_empty_inputs_return_zero():
    X = np.zeros((0, 3))
    Y = np.ones((5, 3))
    assert dist_w2(X, Y)     == 0.0
    assert dist_energy(X, Y) == 0.0