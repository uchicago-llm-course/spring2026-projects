# tests/test_agents.py
import numpy as np
from src.hybrid_bo.core.agents import GPAgent
from src.hybrid_bo.data import MODEL_SPACES, sample_config

def test_gpagent_suggest_returns_valid_config():
    space = MODEL_SPACES["svm"]
    agent = GPAgent(space, seed=0)
    rng   = np.random.RandomState(0)

    for _ in range(3):
        config = sample_config(space, rng)
        agent.observe(config, score=0.9)

    suggestion = agent.suggest()
    for key, (transform, lo, hi) in space.items():
        assert lo <= suggestion[key] <= hi, f"{key}={suggestion[key]} out of [{lo},{hi}]"

def test_gpagent_suggest_before_any_observations():
    space  = MODEL_SPACES["dt"]
    agent  = GPAgent(space, seed=7)
    result = agent.suggest()
    assert set(result.keys()) == set(space.keys())