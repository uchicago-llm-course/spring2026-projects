from unittest.mock import patch
import numpy as np
from src.hybrid_bo.core.experiments import run_single

def _mock_llm_suggest(self):
    """Return a deterministic fallback config without calling OpenAI."""
    return {k: (lo + hi) / 2 for k, (t, lo, hi) in self.space.items()}

@patch("hybrid_bo.core.agents.LLMAgent.suggest", _mock_llm_suggest)
def test_run_single_returns_correct_shapes():
    records, llm_log, switch_result = run_single(
        model_name="dt", dataset="iris",
        n_trials=2, seed=0, start_idx=0,
        engine="gpt-3.5-turbo")

    assert len(records) == 2
    assert len(llm_log) == 2
    assert "w2" in records[0]
    assert "best_gp" in records[0]
    assert switch_result.T == 2