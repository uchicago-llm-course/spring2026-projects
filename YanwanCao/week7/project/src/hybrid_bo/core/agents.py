import json
import numpy as np
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.stats as stats
from openai import OpenAI
from copy import deepcopy

from config import OPENAI_API_KEY
from config.settings import LLM_OUTPUT_PRICE_PER_MILLION, LLM_INPUT_PRICE_PER_MILLION
from src.hybrid_bo.data import warp, unwarp, config_to_warped, warped_to_config


class GPAgent:
    """
    Gaussian-Process Bayesian Optimisation agent.
    State: X_obs, y_obs — only configs proposed by THIS agent.
    """
    def __init__(self, space: dict, seed: int = 0):
        self.space = space
        self.dim   = len(space)
        self.rng   = np.random.RandomState(seed)
        kernel     = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp    = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=seed)
        self.X_obs: list = []
        self.y_obs: list = []

    def observe(self, config: dict, score: float):
        self.X_obs.append(config_to_warped(config, self.space))
        self.y_obs.append(score)

    def suggest(self, n_cand: int = 2000) -> dict:
        X_cand = self.rng.rand(n_cand, self.dim)
        if len(self.X_obs) < 2:
            ei = self.rng.rand(n_cand)
        else:
            self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
            mu, sigma = self.gp.predict(X_cand, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            best  = max(self.y_obs)
            z     = (mu - best) / sigma
            ei    = (mu - best) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return warped_to_config(X_cand[np.argmax(ei)], self.space)


SYSTEM_PROMPT = (
    "You are an expert ML engineer performing Bayesian hyperparameter "
    "optimization. Suggest the single best next configuration to MAXIMIZE "
    "the given scoring metric. "
    "Respond ONLY with a valid JSON object — no explanation, no markdown."
)

class LLMAgent:
    """
    LLM-driven BO agent.
    State: history — only configs proposed by THIS agent + their scores.
    """
    def __init__(self, space: dict, task_desc: str, seed: int = 0,
                 engine: str = "gpt-3.5-turbo"):
        self.space     = space
        self.task_desc = task_desc
        self.engine    = engine
        self.rng       = np.random.RandomState(seed)
        self.history:  list = []
        self.client    = OpenAI(api_key=OPENAI_API_KEY)

    def observe(self, config: dict, score: float, cost: float = 0.0):
        self.history.append({"config": deepcopy(config), "score": score, "cost": cost})

    def _fallback(self) -> dict:
        eps = 0.4
        if len(self.history) == 0 or self.rng.rand() < eps:
            return {k: unwarp(self.rng.rand(), *v) for k, v in self.space.items()}
        best  = max(self.history, key=lambda x: x["score"])["config"]
        noise = max(0.05, 0.3 - 0.01 * len(self.history))
        config = {}
        for k, (t, lo, hi) in self.space.items():
            c = warp(best[k], t, lo, hi)
            u = np.clip(c + self.rng.randn() * noise, 0.01, 0.99)
            config[k] = unwarp(u, t, lo, hi)
        return config

    def suggest(self) -> dict:
        lines = [f"Task: {self.task_desc}", "Search space:"]
        for k, (t, lo, hi) in self.space.items():
            lines.append(f"  {k}: {t}-scale [{lo:.4g}, {hi:.4g}]")
        lines.append("History (best first, this agent only):")
        for obs in sorted(self.history, key=lambda x: -x["score"])[:10]:
            s = ", ".join(f"{k}={v:.4g}" for k, v in obs["config"].items())
            lines.append(f"  score={obs['score']:.5f} | {s}")
        hp_keys = list(self.space.keys())
        lines.append(
            "Respond with JSON only: {"
            + ", ".join(f'"{k}": ...' for k in hp_keys) + "}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.engine,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user",   "content": "\n".join(lines)}],
                temperature=0.0, max_tokens=300,
            )
            input_tokens  = resp.usage.prompt_tokens
            output_tokens = resp.usage.completion_tokens
            cost = (input_tokens  * LLM_INPUT_PRICE_PER_MILLION  / 1_000_000
                    + output_tokens * LLM_OUTPUT_PRICE_PER_MILLION / 1_000_000)
            raw    = resp.choices[0].message.content.strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            config = {}
            for k, (transform, lo, hi) in self.space.items():
                default = (lo * hi) ** 0.5 if transform == "log" else (lo + hi) / 2
                val     = float(parsed.get(k, default))
                config[k] = float(np.clip(val, lo * 1.001, hi * 0.999))
            return config, cost
        except Exception as e:
            print(f"  [LLM fallback] {e}")
            return self._fallback()