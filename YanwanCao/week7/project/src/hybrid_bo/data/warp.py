import numpy as np

def warp(value, transform, lower, upper):
    if transform == "log":
        v = np.log(np.clip(value, lower * 1e-9, upper * 1e9))
        lo, hi = np.log(lower), np.log(upper)
    elif transform == "logit":
        v  = np.log(value / (1 - value))
        lo = np.log(lower / (1 - lower))
        hi = np.log(upper / (1 - upper))
    else:
        v, lo, hi = value, lower, upper
    return (v - lo) / (hi - lo)


def unwarp(u, transform, lower, upper):
    u = float(np.clip(u, 1e-9, 1 - 1e-9))
    if transform == "log":
        lo, hi = np.log(lower), np.log(upper)
        return float(np.exp(lo + u * (hi - lo)))
    if transform == "logit":
        lo = np.log(lower / (1 - lower))
        hi = np.log(upper / (1 - upper))
        z  = lo + u * (hi - lo)
        return float(np.exp(z) / (1 + np.exp(z)))
    return float(lower + u * (upper - lower))


def config_to_warped(config: dict, space: dict) -> np.ndarray:
    return np.array([warp(config[k], *v) for k, v in space.items()])


def warped_to_config(x: np.ndarray, space: dict) -> dict:
    return {k: unwarp(x[i], *v) for i, (k, v) in enumerate(space.items())}


def sample_config(space: dict, rng: np.random.RandomState) -> dict:
    return warped_to_config(rng.rand(len(space)), space)