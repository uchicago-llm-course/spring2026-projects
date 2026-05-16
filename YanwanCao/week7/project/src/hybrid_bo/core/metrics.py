import numpy as np
import ot
from scipy.spatial.distance import cdist


def dist_w2(X: np.ndarray, Y: np.ndarray) -> float:
    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return 0.0

    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric="sqeuclidean")
    return float(np.sqrt(max(ot.emd2(a, b, M), 0.0)))


def dist_mmd(X: np.ndarray, Y: np.ndarray, bandwidth: float = 0.5) -> float:
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    def rbf(A, B):
        return np.exp(-cdist(A, B, "sqeuclidean") / (2 * bandwidth ** 2))
    val = rbf(X, X).mean() - 2 * rbf(X, Y).mean() + rbf(Y, Y).mean()
    return float(np.sqrt(max(val, 0.0)))


def dist_kl(X: np.ndarray, Y: np.ndarray, eps: float = 1e-9) -> float:
    if len(X) < 2 or len(Y) < 2:
        return None
        # return 0.0
    d = X.shape[1]

    def _kl_mvn(mu1, S1, mu2, S2):
        S2i      = np.linalg.inv(S2)
        diff     = mu2 - mu1
        sg1, ld1 = np.linalg.slogdet(S1)
        sg2, ld2 = np.linalg.slogdet(S2)
        if sg1 <= 0 or sg2 <= 0:
            raise np.linalg.LinAlgError
        return 0.5 * (np.trace(S2i @ S1) + diff @ S2i @ diff - d + ld2 - ld1)

    try:
        mu_x  = X.mean(0); cov_x = np.cov(X, rowvar=False) + eps * np.eye(d)
        mu_y  = Y.mean(0); cov_y = np.cov(Y, rowvar=False) + eps * np.eye(d)
        return float(max(_kl_mvn(mu_x, cov_x, mu_y, cov_y)
                        + _kl_mvn(mu_y, cov_y, mu_x, cov_x), 0.0))
    except Exception:
        total = 0.0
        bins  = max(4, min(len(X) // 2, 20))
        for j in range(d):
            lo = min(X[:, j].min(), Y[:, j].min()) - 1e-9
            hi = max(X[:, j].max(), Y[:, j].max()) + 1e-9
            px, _ = np.histogram(X[:, j], bins=bins, range=(lo, hi), density=True)
            py, _ = np.histogram(Y[:, j], bins=bins, range=(lo, hi), density=True)
            px = px + eps; py = py + eps
            px /= px.sum(); py /= py.sum()
            total += float(np.sum(px * np.log(px / py))
                         + np.sum(py * np.log(py / px)))
        return total / d


def dist_energy(X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    exy = cdist(X, Y).mean()
    exx = cdist(X, X).mean() if len(X) > 1 else 0.0
    eyy = cdist(Y, Y).mean() if len(Y) > 1 else 0.0
    return float(max(2 * exy - exx - eyy, 0.0))


def dist_pointwise(X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    return float(np.linalg.norm(X[-1] - Y[-1]))


DISTANCE_METRICS = [
    ("w2",        dist_w2,        "Wasserstein-2",         "W2(LLM||GP)"),
    ("mmd",       dist_mmd,       "Max Mean Discrepancy",  "MMD(LLM||GP)"),
    ("kl",        dist_kl,        "Sym. KL Divergence",    "KL+KL_rev"),
    ("energy",    dist_energy,    "Energy Distance",        "Energy(LLM||GP)"),
    ("pointwise", dist_pointwise, "Pointwise (last step)", "||x_LLM-x_GP||"),
]
DIST_KEYS = [m[0] for m in DISTANCE_METRICS]