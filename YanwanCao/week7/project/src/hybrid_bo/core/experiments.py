import numpy as np
import time
from typing import List

from config.settings import SWITCH_EPSILON, BO_COST_PER_STEP
from src.hybrid_bo.data import (MODEL_SPACES, DISTANCE_METRICS, make_objective, config_to_warped, sample_config)
from src.hybrid_bo.core.agents import LLMAgent, GPAgent
from src.hybrid_bo.core.switching import compute_switch_distances, solve_switching_point, compute_epsilon


def run_single(model_name: str, dataset: str, n_trials: int,
               seed: int, start_idx: int, engine: str,
               llm_delay: float = 0.5):
    """
    One LLM-BO vs GP-BO trajectory pair.

    Both agents run INDEPENDENTLY for all n_trials iterations.
    After the loop, the switching-point MIP is solved using the
    accumulated trajectories to find the optimal tau.

    Returns
    -------
    records       : one dict per trial (distances + scores + HP values)
    llm_log       : LLM-focused subset for a separate CSV
    switch_result : SwitchResult from the Gurobi MIP
    """
    rng     = np.random.RandomState(seed * 1000 + start_idx)
    space   = MODEL_SPACES[model_name]
    hp_keys = list(space.keys())

    _ = sample_config(space, rng)   # consume one slot for reproducibility

    objective, task = make_objective(model_name, dataset)
    metric_label    = "5-fold CV accuracy" if task == "clf" else "5-fold CV neg-MSE"
    task_desc = (
        f"Model: {model_name.upper()} | Dataset: {dataset} | "
        f"Metric: {metric_label}. Search space: "
        + ", ".join(f"{k} ({v[0]}-scale [{v[1]:.3g},{v[2]:.3g}])"
                    for k, v in space.items()) + "."
    )

    # ── Agents are fully independent — no cross-reading of proposals or scores ─
    gp_agent  = GPAgent(space, seed=seed * 1000 + start_idx)
    llm_agent = LLMAgent(space, task_desc,
                         seed=seed * 1000 + start_idx, engine=engine)

    # Trajectory lists (warped HP arrays) — used for distance metrics AND
    # for the switching-point constraint.
    gp_traj:  List[np.ndarray] = []
    llm_traj: List[np.ndarray] = []

    records: list = []
    llm_log: list = []

    sep = "-" * 82
    print(f"\n  {sep}")
    print(f"  MODEL={model_name.upper()}  DATASET={dataset}  "
          f"seed={seed}  start={start_idx}  task={task}")
    print(f"  {sep}")


    llm_costs = []
    for t in range(1, n_trials + 1):

        # ── GP agent: suggest → evaluate → observe (self-contained) ──────────
        gc = gp_agent.suggest()
        gs = objective(gc)
        gp_agent.observe(gc, gs)

        # ── LLM agent: suggest → evaluate → observe (self-contained) ─────────
        # LLMAgent.suggest() builds its prompt from self.history (t-1 entries at
        # trial t), then calls the OpenAI API.  self.history grows only with
        # LLM-proposed configs — it never sees GP proposals or GP scores.
        time.sleep(llm_delay)
        lc, llm_cost = llm_agent.suggest()
        ls = objective(lc)
        llm_agent.observe(lc, ls, llm_cost)
        llm_costs.append(llm_cost)

        # ── Record warped HP vectors for trajectory distance metrics ──────────
        gp_traj.append(config_to_warped(gc, space))
        llm_traj.append(config_to_warped(lc, space))

        X_llm = np.array(llm_traj)
        X_gp  = np.array(gp_traj)

        # All 5 distribution-level distance metrics (cumulative trajectories)
      #   dists = {key: fn(X_llm, X_gp) for key, fn, *_ in DISTANCE_METRICS}
        dists = {
                  key: (fn(X_llm, X_gp) if not (key == "kl" and (len(X_llm) < 2 or len(X_gp) < 2)) else None)
                  for key, fn, *_ in DISTANCE_METRICS
                  }
        best_gp  = max(gp_agent.y_obs)
        best_llm = max(o["score"] for o in llm_agent.history)
        is_new   = (ls == best_llm)

        print(f"\n  [t={t:03d}] LLM-> " +
              " | ".join(f"{k}={lc[k]:.4g}" for k in hp_keys) +
              f"  score={ls:.5f}{'  ** NEW BEST **' if is_new else ''}")
        print(f"         GP  -> " +
              " | ".join(f"{k}={gc[k]:.4g}" for k in hp_keys) +
              f"  score={gs:.5f}")
      #   print("         dists: " +
      #         "  ".join(f"{key}={v:.4f}" for key, v in dists.items()))
        print("         dists: " +
              "  ".join(f"{key}={v:.4f}" if v is not None else f"{key}=N/A"
                    for key, v in dists.items()))
        row: dict = dict(
            model=model_name, dataset=dataset,
            trial=t, seed=seed, start_idx=start_idx,
            **dists,
            llm_cost = llm_cost,
            gp_score_t=gs, llm_score_t=ls,
            best_gp=best_gp, best_llm=best_llm,
            **{f"gp_{k}":  gc[k] for k in hp_keys},
            **{f"llm_{k}": lc[k] for k in hp_keys},
        )
        records.append(row)

        llm_log.append(dict(
            model=model_name, dataset=dataset,
            trial=t, seed=seed, start_idx=start_idx,
            llm_score_t=ls, best_llm=best_llm, best_gp=best_gp,
            is_new_best=int(is_new),
            **dists,
            **{f"llm_{k}": lc[k] for k in hp_keys},
        ))

    # ─────────────────────────────────────────────────────────────────────────
    # SWITCHING-POINT OPTIMISATION
    # Runs AFTER the trial loop.  Uses the completed llm_traj / gp_traj to
    # compute per-trial pointwise distances, then solves the Gurobi MIP to
    # find the optimal tau minimising total cost subject to the alignment
    # constraint.
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  DONE  model={model_name}  dataset={dataset}  "
          f"seed={seed}  start={start_idx}  "
          f"best_llm={max(o['score'] for o in llm_agent.history):.5f}  "
          f"best_gp={max(gp_agent.y_obs):.5f}")
    print(f"  Running switching-point MIP (T={n_trials}, epsilon={SWITCH_EPSILON}) ...")

    # cost_llm, cost_bo = build_cost_trajectories(n_trials)
    # sw_distances      = compute_switch_distances(llm_traj, gp_traj)
    # cost_llm, cost_bo = build_cost_trajectories(n_trials)
    cost_llm = llm_costs
    cost_bo  = [BO_COST_PER_STEP] * n_trials
    sw_distances      = compute_switch_distances(llm_traj, gp_traj)
    epsilon           = compute_epsilon(sw_distances)   # ← dynamic per run
    switch_result = solve_switching_point(
        T          = n_trials,
        cost_llm   = cost_llm,
        cost_bo    = cost_bo,
        distances  = sw_distances,
        epsilon    = epsilon,
        model_name = model_name,
        dataset    = dataset,
        seed       = seed,
        start_idx  = start_idx,
    )

    if switch_result.feasible:
        print(f"  [switch] optimal tau={switch_result.tau}  "
              f"total_cost=${switch_result.total_cost:.6f}  "
              f"avg_dist@tau={switch_result.avg_dist_at_tau:.4f}")
    else:
        print("  [switch] INFEASIBLE — no tau satisfies the alignment constraint "
              f"(epsilon={SWITCH_EPSILON}).")

    return records, llm_log, switch_result

