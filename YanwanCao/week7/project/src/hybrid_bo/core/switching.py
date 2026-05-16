from dataclasses import dataclass
from typing import List
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from config.settings import SWITCH_EPSILON_FRACTION
from config.settings import BO_COST_PER_STEP, LLM_INPUT_PRICE_PER_MILLION, LLM_OUTPUT_PRICE_PER_MILLION, BASE_INPUT_TOKENS, TOKENS_PER_HISTORY, OUTPUT_TOKENS


@dataclass
class SwitchResult:
    """Result of the switching-point MIP for one (seed, start) run."""
    tau:             int    # optimal switching trial (1-indexed); -1 = infeasible
    total_cost:      float
    llm_phase_cost:  float  # sum_{t=1}^{tau}   cost_LLM(t)
    bo_phase_cost:   float  # sum_{t=tau+1}^{T} cost_BO(t)
    avg_dist_at_tau: float  # (1/tau) sum_{t=1}^{tau} ||LLM_t - BO_t||
    feasible:        bool
    model_name:      str  = ""
    dataset:         str  = ""
    seed:            int  = 0
    start_idx:       int  = 0
    T:               int  = 0
    epsilon:         float = 0.0


def compute_epsilon(distances: List[float], fraction: float = SWITCH_EPSILON_FRACTION) -> float:
    """
    Set epsilon = fraction * max(distances).
    At fraction=0.40: switch only after avg distance has dropped
    to 40% of its worst-case value — i.e. 60% convergence achieved.
    """
    if not distances:
        return 1.0
    # max_dist = max(distances)
    # epsilon  = fraction * max_dist
    # print(f"  [epsilon] max_dist={max_dist:.5f}  fraction={fraction:.2f}  "
    #       f"→  epsilon={epsilon:.5f}")
    # return epsilon
    d        = np.asarray(distances)
    avg_dist = np.cumsum(d) / np.arange(1, len(d) + 1)  # running avg, same as MIP uses
    min_dist = float(avg_dist.min())
    max_dist = float(avg_dist.max())
    epsilon  = min_dist + fraction * (max_dist - min_dist)
    # print(f"  [epsilon] avg_dist range=[{min_dist:.5f}, {max_dist:.5f}]  "
    #       f"fraction={fraction:.2f}  →  epsilon={epsilon:.5f}")
    return epsilon


def solve_switching_point(
    T:          int,
    cost_llm:   List[float],
    cost_bo:    List[float],
    distances:  List[float],
    epsilon:    float = 0.0,
    model_name: str   = "",
    dataset:    str   = "",
    seed:       int   = 0,
    start_idx:  int   = 0,
    verbose:    bool  = False,
) -> SwitchResult:
    """
    Solve the switching-point MIP via Gurobi.

    Formulation
    -----------
    Variables : z[i] ∈ {0,1}  for i = 0..T-1  (i = tau-1 in 0-indexed)

    Constraints:
        sum_i z[i] = 1                             (exactly one tau)
        sum_i avg_dist[i] * z[i] <= epsilon - 1e-9 (strict alignment)

    Objective:
        min  sum_i obj_coeff[i] * z[i]

    where:
        cumllm[i]    = sum_{t=1}^{i+1}  cost_llm[t-1]   (prefix sum, LLM up to tau)
        cumbo[i]     = sum_{t=1}^{i+1}  cost_bo[t-1]
        obj_coeff[i] = cumllm[i] + (total_bo - cumbo[i]) (LLM phase + BO phase)
        avg_dist[i]  = (1/(i+1)) * sum_{t=1}^{i+1} dist[t-1]

    This is a pure binary LP solved exactly by Gurobi in milliseconds.
    """

    cl = np.asarray(cost_llm)
    cb = np.asarray(cost_bo)
    d  = np.asarray(distances)

    cumllm   = np.cumsum(cl)
    cumbo    = np.cumsum(cb)
    cumdist  = np.cumsum(d)
    total_bo = float(cumbo[-1])

    obj_coeff = cumllm + (total_bo - cumbo)
    avg_dist  = cumdist / np.arange(1, T + 1)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1 if verbose else 0)
        env.start()
        with gp.Model(env=env, name="llm_bo_switch") as m:

            z = m.addVars(T, vtype=GRB.BINARY, name="z")

            m.addConstr(
                gp.quicksum(z[i] for i in range(T)) == 1,
                name="one_switch",
            )
            m.addConstr(
                gp.quicksum(avg_dist[i] * z[i] for i in range(T))
                    <= epsilon - 1e-9,
                name="alignment",
            )
            m.setObjective(
                gp.quicksum(obj_coeff[i] * z[i] for i in range(T)),
                GRB.MINIMIZE,
            )
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                idx = int(np.argmax([z[i].X for i in range(T)]))
                tau = idx + 1
                return SwitchResult(
                    tau            = tau,
                    total_cost     = float(m.ObjVal),
                    llm_phase_cost = float(cumllm[idx]),
                    bo_phase_cost  = float(total_bo - cumbo[idx]),
                    avg_dist_at_tau= float(avg_dist[idx]),
                    feasible       = True,
                    model_name     = model_name,
                    dataset        = dataset,
                    seed           = seed,
                    start_idx      = start_idx,
                    T              = T,
                    epsilon        = epsilon,
                )
            else:
                return SwitchResult(
                    tau=-1, total_cost=float("inf"),
                    llm_phase_cost=float("inf"), bo_phase_cost=float("inf"),
                    avg_dist_at_tau=float("inf"), feasible=False,
                    model_name=model_name, dataset=dataset,
                    seed=seed, start_idx=start_idx, T=T, epsilon=epsilon,
                )


def summarise_switch_results(
    results:    List[SwitchResult],
    model_name: str = "",
    dataset:    str = "",
    output_dir: str = "results",
) -> pd.DataFrame:
    """Aggregate SwitchResult objects from all (seed, start) runs and save CSV."""
    rows = [{
        "model":          r.model_name,
        "dataset":        r.dataset,
        "seed":           r.seed,
        "start_idx":      r.start_idx,
        "T":              r.T,
        "epsilon":        r.epsilon,
        "tau":            r.tau,
        "feasible":       r.feasible,
        "total_cost":     r.total_cost,
        "llm_phase_cost": r.llm_phase_cost,
        "bo_phase_cost":  r.bo_phase_cost,
        "avg_dist_at_tau":r.avg_dist_at_tau,
    } for r in results]

    df    = pd.DataFrame(rows)
    feas  = df[df["feasible"]]

    print(f"\n{'='*60}")
    print(f"  SWITCH-POINT SUMMARY  model={model_name}  dataset={dataset}")
    print(f"{'='*60}")
    print(f"  Total runs   : {len(df)}")
    print(f"  Feasible     : {len(feas)}  ({100*len(feas)/max(len(df),1):.0f}%)")
    if len(feas):
        print(f"  tau  mean={feas['tau'].mean():.1f}  "
              f"std={feas['tau'].std():.1f}  "
              f"min={feas['tau'].min()}  max={feas['tau'].max()}")
        print(f"  total cost   : ${feas['total_cost'].mean():.6f}  (mean)")
        print(f"  LLM phase    : ${feas['llm_phase_cost'].mean():.6f}  (mean)")
        print(f"  BO  phase    : ${feas['bo_phase_cost'].mean():.6f}  (mean)")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}_{dataset}_switch_results.csv")
    df.to_csv(path, index=False)
    print(f"[saved switch results] {path}")
    return df

# def _llm_cost_at_trial(t: int) -> float:
#     """
#     USD cost of the LLM call at trial t (1-indexed).

#     Matches LLMAgent.suggest(): at trial t the history has t-1 entries,
#     each adding TOKENS_PER_HISTORY input tokens to the prompt.
#     """
#     input_tokens = BASE_INPUT_TOKENS + (t - 1) * TOKENS_PER_HISTORY
#     return (input_tokens  * LLM_INPUT_PRICE_PER_MILLION  / 1_000_000
#           + OUTPUT_TOKENS * LLM_OUTPUT_PRICE_PER_MILLION / 1_000_000)


# def build_cost_trajectories(T: int):
#     """
#     Return (cost_llm, cost_bo) — lists of length T.
#     cost_llm[t-1] = USD cost of LLM call at trial t.
#     cost_bo[t-1]  = $0.
#     """
#     cost_llm = [_llm_cost_at_trial(t) for t in range(1, T + 1)]
#     cost_bo  = [BO_COST_PER_STEP] * T
#     return cost_llm, cost_bo


def compute_switch_distances(
    llm_traj: List[np.ndarray],
    bo_traj:  List[np.ndarray],
) -> List[float]:
    """
    Pointwise Euclidean distances in warped HP space for the constraint.
    distances[t-1] = ||llm_traj[t-1] - bo_traj[t-1]||_2
    """
    return [
        float(np.linalg.norm(np.asarray(l) - np.asarray(b)))
        for l, b in zip(llm_traj, bo_traj)
    ]


def plot_switch_results(
    df_switch:  pd.DataFrame,
    model_name: str,
    dataset:    str,
    output_dir: str = "results",
    T:          int = 25,
):
    """Tau histogram + per-run cost breakdown bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    feas = df_switch[df_switch["feasible"]]
    if feas.empty:
        print(f"[plot_switch_results] No feasible runs for {model_name}/{dataset}.")
        return

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Switching-point optimisation — {model_name.upper()} / {dataset}\n"
        f"epsilon={feas['epsilon'].iloc[0]:.2f}   "
        f"feasible: {len(feas)}/{len(df_switch)} runs",
        fontsize=11,
    )

    ax_hist.hist(feas["tau"], bins=range(1, T + 2),
                 color="#4363d8", edgecolor="white", alpha=0.85)
    ax_hist.axvline(feas["tau"].mean(), color="red", lw=2, ls="--",
                    label=f"mean tau = {feas['tau'].mean():.1f}")
    ax_hist.set(
        title="Distribution of optimal tau",
        xlabel="tau  [use LLM for t=1..tau, GP for t=tau+1..T]",
        ylabel="Count", xlim=(0, T + 1),
    )
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    labels = [f"s{int(r.seed)}/r{int(r.start_idx)}" for _, r in feas.iterrows()]
    x = np.arange(len(labels))
    ax_bar.bar(x, feas["llm_phase_cost"].values,
               label="LLM phase cost ($)", color="#e6194b", alpha=0.85)
    ax_bar.bar(x, feas["bo_phase_cost"].values,
               bottom=feas["llm_phase_cost"].values,
               label="BO phase cost ($0)", color="#3cb44b", alpha=0.85)
    ax_bar.set(
        title="Cost breakdown per run (LLM + BO phases)",
        xlabel="(seed / start_idx)", ylabel="USD",
        xticks=x, xticklabels=labels,
    )
    ax_bar.tick_params(axis="x", labelsize=7, rotation=45)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(output_dir, f"{model_name}_{dataset}_switch_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved switch plot] {out}")
    plt.close()