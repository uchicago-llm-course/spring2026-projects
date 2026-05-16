import numpy as np
import pytest

@pytest.fixture
def small_rng():
    return np.random.RandomState(42)

@pytest.fixture
def iris_data():
    from sklearn.datasets import load_iris
    d = load_iris()
    return d.data, d.target

@pytest.fixture
def svm_space():
    from src.hybrid_bo.data import MODEL_SPACES
    return MODEL_SPACES["svm"]