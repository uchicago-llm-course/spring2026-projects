import argparse
import os
from typing import List, Optional
import numpy as np
import pandas as pd



ALL_MODELS   = ["svm", "dt", "rf", "mlp", "adaboost"]
ALL_DATASETS = ["iris", "wine", "digits", "breast", "diabetes", "boston"]


def cost_pure_llm_from_df(traj_run: pd.DataFrame) -> float:
    return float(traj_run["llm_cost"].sum())

def cost_switching_from_df(traj_run: pd.DataFrame, tau: int) -> float:
    return float(traj_run[traj_run["trial"] <= tau]["llm_cost"].sum())


# =============================================================================
#  Performance: first trial to reach threshold of final best score
# =============================================================================

def trials_to_target(
    best_scores: np.ndarray,   # shape (T,), running best per trial
    threshold:   float,         # absolute score value to reach
) -> int:
    """Return first trial index (1-indexed) where best_scores >= threshold, else T."""
    hits = np.where(best_scores >= threshold)[0]
    return int(hits[0] + 1) if len(hits) else len(best_scores)


def switching_trajectory(
    best_llm: np.ndarray,   # shape (T,)
    best_gp:  np.ndarray,   # shape (T,)
    tau:      int,           # 1-indexed switching trial
) -> np.ndarray:
    """
    Construct the running-best trajectory for the switching strategy:
      t <= tau : follows best_llm(t)
      t > tau  : follows max(best_llm(tau), best_gp(t))
                 because after switching we keep the best LLM result found
                 so far and let GP improve from there.
    """
    T    = len(best_llm)
    traj = np.empty(T)
    for t in range(T):
        if t + 1 <= tau:
            traj[t] = best_llm[t]
        else:
            traj[t] = max(best_llm[tau - 1], best_gp[t])
    return traj


# =============================================================================
#  Per-(seed, start) ablation row
# =============================================================================

def ablation_one_run(
    traj_run:   pd.DataFrame,   # rows for one (seed, start_idx), sorted by trial
    tau:        int,
    T:          int,
    threshold_pct: float = 0.95,
) -> dict:
    """
    Compute all ablation metrics for a single (seed, start) run.

    Parameters
    ----------
    traj_run      : subset of trajectories CSV for this (seed, start)
    tau           : optimal switching trial from switch_results CSV
    T             : total number of trials
    threshold_pct : fraction of best final score used as performance target
    """
    best_llm = traj_run.sort_values("trial")["best_llm"].values   # shape (T,)
    best_gp  = traj_run.sort_values("trial")["best_gp"].values    # shape (T,)

    # ── Costs ────────────────────────────────────────────────────────────────
    c_pure   = cost_pure_llm_from_df(traj_run)
    c_switch = cost_switching_from_df(traj_run, tau)
    saving   = c_pure - c_switch
    saving_pct = 100.0 * saving / c_pure if c_pure > 0 else 0.0

    # ── Performance threshold ─────────────────────────────────────────────────
    # Use the best score ever seen by either agent as the reference ceiling.
    # This is a fair upper bound — neither agent "wins" by definition.
    best_final = max(best_llm[-1], best_gp[-1])
    threshold  = threshold_pct * best_final

    sw_traj = switching_trajectory(best_llm, best_gp, tau)

    llm_t2t    = trials_to_target(best_llm, threshold)
    switch_t2t = trials_to_target(sw_traj,  threshold)
    trials_saved = llm_t2t - switch_t2t   # positive = switching is faster

    # ── Behaviour indicators ──────────────────────────────────────────────────
    # How much did LLM improve AFTER tau?  If ≈ 0, switching costs no quality.
    llm_improvement_after_tau = float(best_llm[-1] - best_llm[min(tau - 1, T - 1)])
    # How much better is GP's final result vs LLM's?
    gp_advantage = float(best_gp[-1] - best_llm[-1])

    # ── Quality of switching vs pure LLM at trial T ───────────────────────────
    switch_final_score = float(sw_traj[-1])
    llm_final_score    = float(best_llm[-1])
    gp_final_score     = float(best_gp[-1])
    quality_delta      = switch_final_score - llm_final_score  # >= 0 if GP helps

    return dict(
        tau                      = tau,
        T                        = T,
        cost_pure_llm_usd        = c_pure,
        cost_switching_usd       = c_switch,
        cost_saving_usd          = saving,
        cost_saving_pct          = saving_pct,
        threshold_pct            = threshold_pct,
        threshold_score          = threshold,
        llm_trials_to_target     = llm_t2t,
        switch_trials_to_target  = switch_t2t,
        trials_saved             = trials_saved,
        llm_final_score          = llm_final_score,
        gp_final_score           = gp_final_score,
        switch_final_score       = switch_final_score,
        quality_delta            = quality_delta,      # switch - pure LLM
        gp_advantage             = gp_advantage,       # GP - LLM at T
        llm_improvement_after_tau= llm_improvement_after_tau,
        llm_plateaued            = llm_improvement_after_tau < 0.001,
    )


# =============================================================================
#  Per-(model, dataset) ablation
# =============================================================================

def ablation_model_dataset(
    traj_df:    pd.DataFrame,
    switch_df:  pd.DataFrame,
    model_name: str,
    dataset:    str,
    threshold_pct: float = 0.95,
) -> pd.DataFrame:
    """
    Run ablation for every feasible (seed, start_idx) pair and return a
    DataFrame with one row per run + a summary row.
    """
    feasible = switch_df[switch_df["feasible"] == True].copy()
    if feasible.empty:
        print(f"  [ablation] No feasible tau for {model_name}/{dataset} — skipping.")
        return pd.DataFrame()

    T = int(switch_df["T"].iloc[0])
    rows = []

    for _, sw_row in feasible.iterrows():
        seed      = int(sw_row["seed"])
        start_idx = int(sw_row["start_idx"])
        tau       = int(sw_row["tau"])

        run_df = traj_df[
            (traj_df["seed"]      == seed) &
            (traj_df["start_idx"] == start_idx)
        ]
        if run_df.empty:
            continue

        metrics = ablation_one_run(run_df, tau, T, threshold_pct)
        metrics.update(dict(
            model     = model_name,
            dataset   = dataset,
            seed      = seed,
            start_idx = start_idx,
        ))
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Print per-run table ───────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  ABLATION: {model_name.upper()} / {dataset}   "
          f"(threshold={threshold_pct*100:.0f}% of best final score)")
    print(f"{'─'*72}")
    print(f"  {'seed':>4} {'start':>5} {'tau':>4} {'cost_LLM$':>10} "
          f"{'cost_sw$':>10} {'saving%':>8} "
          f"{'LLM_t2t':>8} {'sw_t2t':>7} {'t_saved':>8} "
          f"{'LLM_fin':>8} {'GP_fin':>7} {'sw_fin':>7} {'plateaued':>10}")
    print(f"  {'─'*4:>4} {'─'*5:>5} {'─'*4:>4} {'─'*10:>10} "
          f"{'─'*10:>10} {'─'*8:>8} "
          f"{'─'*8:>8} {'─'*7:>7} {'─'*8:>8} "
          f"{'─'*8:>8} {'─'*7:>7} {'─'*7:>7} {'─'*10:>10}")
    for r in rows:
        print(f"  {r['seed']:>4} {r['start_idx']:>5} {r['tau']:>4} "
              f"  {r['cost_pure_llm_usd']:.6f} "
              f"  {r['cost_switching_usd']:.6f} "
              f"  {r['cost_saving_pct']:>7.1f}% "
              f"  {r['llm_trials_to_target']:>7} "
              f"  {r['switch_trials_to_target']:>6} "
              f"  {r['trials_saved']:>+7} "
              f"  {r['llm_final_score']:>8.5f} "
              f"  {r['gp_final_score']:>7.5f} "
              f"  {r['switch_final_score']:>7.5f} "
              f"  {'YES' if r['llm_plateaued'] else 'no':>10}")

    # ── Summary statistics ────────────────────────────────────────────────────
    num_cols = [
        "tau", "cost_pure_llm_usd", "cost_switching_usd",
        "cost_saving_usd", "cost_saving_pct",
        "llm_trials_to_target", "switch_trials_to_target", "trials_saved",
        "llm_final_score", "gp_final_score", "switch_final_score",
        "quality_delta", "gp_advantage", "llm_improvement_after_tau",
    ]
    means = df[num_cols].mean()
    stds  = df[num_cols].std().fillna(0)

    print(f"\n  ── MEAN ± STD across {len(df)} feasible runs ──")
    print(f"  tau                     : {means['tau']:.1f} ± {stds['tau']:.1f}")
    print(f"  cost_pure_llm           : ${means['cost_pure_llm_usd']:.6f} ± {stds['cost_pure_llm_usd']:.6f}")
    print(f"  cost_switching          : ${means['cost_switching_usd']:.6f} ± {stds['cost_switching_usd']:.6f}")
    print(f"  cost_saving             : ${means['cost_saving_usd']:.6f} ({means['cost_saving_pct']:.1f}%)")
    print(f"  trials_to_target (LLM)  : {means['llm_trials_to_target']:.1f} ± {stds['llm_trials_to_target']:.1f}")
    print(f"  trials_to_target (sw)   : {means['switch_trials_to_target']:.1f} ± {stds['switch_trials_to_target']:.1f}")
    print(f"  trials_saved            : {means['trials_saved']:+.1f} ± {stds['trials_saved']:.1f}")
    print(f"  llm_final_score         : {means['llm_final_score']:.5f} ± {stds['llm_final_score']:.5f}")
    print(f"  gp_final_score          : {means['gp_final_score']:.5f} ± {stds['gp_final_score']:.5f}")
    print(f"  switch_final_score      : {means['switch_final_score']:.5f} ± {stds['switch_final_score']:.5f}")
    print(f"  quality_delta (sw-LLM)  : {means['quality_delta']:+.5f}  "
          f"({'GP helps' if means['quality_delta'] > 0.001 else 'negligible'})")
    print(f"  gp_advantage (GP-LLM)   : {means['gp_advantage']:+.5f}  "
          f"({'GP dominates' if means['gp_advantage'] > 0.005 else 'similar'})")
    print(f"  LLM improvement >tau    : {means['llm_improvement_after_tau']:+.5f}  "
          f"({'LLM still learning' if means['llm_improvement_after_tau'] > 0.001 else 'LLM plateaued'})")
    plateaued_pct = 100 * df["llm_plateaued"].mean()
    print(f"  LLM plateaued at tau    : {plateaued_pct:.0f}% of runs")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n  ── INTERPRETATION ──")
    if means["cost_saving_pct"] > 5:
        print(f"  ✓ Switching saves {means['cost_saving_pct']:.1f}% of LLM API cost on average.")
    else:
        print(f"  ~ Cost saving is small ({means['cost_saving_pct']:.1f}%), "
              f"tau is near T so LLM runs most trials anyway.")

    if means["trials_saved"] > 0:
        print(f"  ✓ Switching reaches the {threshold_pct*100:.0f}% performance target "
              f"{means['trials_saved']:.1f} trials earlier than pure LLM.")
    elif means["trials_saved"] < 0:
        print(f"  ✗ Switching reaches the target {abs(means['trials_saved']):.1f} trials "
              f"LATER — LLM finds good configs early in this setting.")
    else:
        print(f"  ~ Both strategies reach the target at the same trial.")

    if means["gp_advantage"] > 0.005:
        print(f"  ✓ GP achieves {means['gp_advantage']:+.5f} higher final score — "
              f"switching also improves solution quality.")
    elif means["gp_advantage"] < -0.005:
        print(f"  ✗ LLM achieves higher final score than GP here — "
              f"switching may trade quality for cost.")
    else:
        print(f"  ~ LLM and GP reach similar final scores; "
              f"switching is cost-efficient without quality loss.")

    if plateaued_pct > 50:
        print(f"  ✓ LLM plateaued in {plateaued_pct:.0f}% of runs before T — "
              f"tau is well-chosen; late LLM calls add cost but no gain.")

    return df


# =============================================================================
#  Plot: cost saving + trials-to-target heatmaps across models × datasets
# =============================================================================

def plot_ablation_heatmaps(
    summary_df: pd.DataFrame,
    output_dir: str = "results",
    threshold_pct: float = 0.95,
):
    """
    Two heatmaps (models × datasets):
        Left  — mean cost saving (%)
        Right — mean trials saved to reach threshold
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    models   = [m for m in ALL_MODELS   if m in summary_df["model"].unique()]
    datasets = [d for d in ALL_DATASETS if d in summary_df["dataset"].unique()]

    def _pivot(col):
        return (summary_df.groupby(["model", "dataset"])[col]
                .mean().unstack("dataset")
                .reindex(index=models, columns=datasets))

    cost_piv   = _pivot("cost_saving_pct")
    trials_piv = _pivot("trials_saved")

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2 * len(datasets) + 2),
                                             max(4, len(models) + 1)))
    fig.suptitle(
        f"Ablation: Pure LLM vs Switching  "
        f"(threshold={threshold_pct*100:.0f}% of best final score)",
        fontsize=12,
    )

    def _heatmap(ax, data, title, fmt, cmap, center=None):
        import matplotlib.cm as cm
        vals = data.values.astype(float)
        if center is not None:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            # norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(vals),
            #                              vcenter=center,
            #                              vmax=np.nanmax(vals))
            if vmin < center < vmax:
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
        else:
            im = ax.imshow(vals, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_yticks(range(len(models)));   ax.set_yticklabels(models)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(models)):
            for j in range(len(datasets)):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=8, color="black")

    _heatmap(axes[0], cost_piv,
             "Mean cost saving (%)\n(higher = switching saves more)",
             "{:.1f}%", "YlGn")
    _heatmap(axes[1], trials_piv,
             f"Mean trials saved to {threshold_pct*100:.0f}% target\n"
             "(positive = switching is faster)",
             "{:+.1f}", "RdYlGn", center=0)

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f"ablation_heatmaps_thr{int(threshold_pct*100)}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[saved heatmaps] {out}")
    plt.close()


def plot_ablation_trajectories(
    traj_df:    pd.DataFrame,
    switch_df:  pd.DataFrame,
    model_name: str,
    dataset:    str,
    output_dir: str = "results",
    threshold_pct: float = 0.95,
):
    """
    Per-(model, dataset) figure with 3 panels:
        Left   — running best score: pure LLM vs switching vs pure GP
        Centre — per-trial LLM cost (token cost grows with t)
        Right  — cumulative cost: pure LLM vs switching
    Uses the mean tau across feasible runs.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    feasible = switch_df[switch_df["feasible"] == True]
    if feasible.empty:
        return

    T         = int(switch_df["T"].iloc[0])
    mean_tau  = int(round(feasible["tau"].mean()))
    trials    = np.arange(1, T + 1)

    # Aggregate across all feasible (seed, start) runs
    grouped  = traj_df.groupby("trial")
    mu_llm   = grouped["best_llm"].mean().values
    mu_gp    = grouped["best_gp"].mean().values
    std_llm  = grouped["best_llm"].std().fillna(0).values
    std_gp   = grouped["best_gp"].std().fillna(0).values

    mu_sw    = switching_trajectory(mu_llm, mu_gp, mean_tau)

    best_final = max(mu_llm[-1], mu_gp[-1])
    threshold  = threshold_pct * best_final

    mean_cost_by_trial = traj_df.groupby("trial")["llm_cost"].mean()

    per_trial_cost = [mean_cost_by_trial[t] for t in trials]  # for centre panel
    cum_llm        = np.cumsum(per_trial_cost)
    cum_sw_correct = np.array([
        sum(mean_cost_by_trial[s] for s in range(1, min(t, mean_tau) + 1))
        for t in trials
    ])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Ablation trajectories — {model_name.upper()} / {dataset}\n"
        f"mean tau={mean_tau}  (avg over {len(feasible)} feasible runs)  |  "
        f"threshold={threshold_pct*100:.0f}% = {threshold:.5f}",
        fontsize=10,
    )

    # ── Left: score trajectories ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(trials, mu_llm, color="#e6194b", lw=2,  label="Pure LLM")
    ax.plot(trials, mu_gp,  color="#3cb44b", lw=2,  label="Pure GP",  ls="--")
    ax.plot(trials, mu_sw,  color="#4363d8", lw=2,  label=f"Switching (tau={mean_tau})", ls="-.")
    ax.fill_between(trials, mu_llm - std_llm, mu_llm + std_llm, alpha=0.12, color="#e6194b")
    ax.fill_between(trials, mu_gp  - std_gp,  mu_gp  + std_gp,  alpha=0.12, color="#3cb44b")
    ax.axvline(mean_tau, color="grey", lw=1.2, ls=":", label=f"tau={mean_tau}")
    ax.axhline(threshold, color="orange", lw=1.2, ls="--", label=f"{threshold_pct*100:.0f}% threshold")

    # Mark trials-to-target
    llm_t2t = trials_to_target(mu_llm, threshold)
    sw_t2t  = trials_to_target(mu_sw,  threshold)
    if llm_t2t <= T:
        ax.scatter([llm_t2t], [mu_llm[llm_t2t - 1]], color="#e6194b", zorder=5, s=60)
    if sw_t2t <= T:
        ax.scatter([sw_t2t],  [mu_sw[sw_t2t  - 1]],  color="#4363d8", zorder=5, s=60)

    ax.set(title="Running best score (mean ± std)",
           xlabel="Trial t", ylabel="Best CV score")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Centre: per-trial LLM token cost ─────────────────────────────────────
    ax = axes[1]
    per_trial_cost = [mean_cost_by_trial[t] for t in trials]
    ax.bar(trials, per_trial_cost, color="#4363d8", alpha=0.75, width=0.8)
    ax.axvline(mean_tau, color="red", lw=1.5, ls="--", label=f"tau={mean_tau}")
    ax.set(title="Per-trial LLM API cost\n(grows as prompt history accumulates)",
           xlabel="Trial t", ylabel="USD per call")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Right: cumulative cost ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(trials, cum_llm,         color="#e6194b", lw=2,   label="Pure LLM (cumulative)")
    ax.plot(trials, cum_sw_correct,  color="#4363d8", lw=2,   label="Switching (cumulative)", ls="-.")
    ax.axvline(mean_tau, color="grey", lw=1.2, ls=":", label=f"tau={mean_tau}")
    total_saving = cum_llm[-1] - cum_sw_correct[-1]
    ax.annotate(
        f"Saving\n${total_saving:.5f}\n({100*total_saving/cum_llm[-1]:.1f}%)",
        xy=(T, cum_llm[-1]),
        xytext=(-60, -30), textcoords="offset points",
        fontsize=8, color="#e6194b",
        arrowprops=dict(arrowstyle="->", color="#e6194b"),
    )
    ax.set(title="Cumulative LLM API cost\n(switching stops paying after tau)",
           xlabel="Trial t", ylabel="Cumulative USD")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir,
                       f"{model_name}_{dataset}_ablation_trajectories.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved ablation trajectories] {out}")
    plt.close()


# =============================================================================
#  Main entry point
# =============================================================================

def run_ablation(
    output_dir:    str   = "results",
    models:        Optional[List[str]] = None,
    datasets:      Optional[List[str]] = None,
    threshold_pct: float = 0.95,
):
    models   = models   or ALL_MODELS
    datasets = datasets or ALL_DATASETS

    all_ablation_rows = []

    for mdl in models:
        for ds in datasets:
            traj_path   = os.path.join(output_dir, f"{mdl}_{ds}_trajectories.csv")
            switch_path = os.path.join(output_dir, f"{mdl}_{ds}_switch_results.csv")

            if not os.path.exists(traj_path):
                print(f"  [skip] {traj_path} not found.")
                continue
            if not os.path.exists(switch_path):
                print(f"  [skip] {switch_path} not found.")
                continue

            traj_df   = pd.read_csv(traj_path)
            switch_df = pd.read_csv(switch_path)

            # Per-(seed, start) ablation
            abl_df = ablation_model_dataset(
                traj_df, switch_df, mdl, ds, threshold_pct)

            if not abl_df.empty:
                all_ablation_rows.append(abl_df)
                # Per-(model, dataset) trajectory plot
                plot_ablation_trajectories(
                    traj_df, switch_df, mdl, ds, output_dir, threshold_pct)

    if not all_ablation_rows:
        print("\nNo results found. Run hybrid_bo_experiment.py first.")
        return

    # ── Global summary CSV ────────────────────────────────────────────────────
    summary_df = pd.concat(all_ablation_rows, ignore_index=True)
    out_csv    = os.path.join(output_dir,
                              f"ablation_summary_thr{int(threshold_pct*100)}.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[saved ablation summary CSV] {out_csv}")

    # ── Cross-model/dataset heatmaps ──────────────────────────────────────────
    plot_ablation_heatmaps(summary_df, output_dir, threshold_pct)

    # ── Global summary table ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  GLOBAL ABLATION SUMMARY  (threshold={threshold_pct*100:.0f}%)")
    print(f"{'='*72}")
    global_means = summary_df[[
        "cost_saving_pct", "trials_saved",
        "llm_final_score", "gp_final_score", "switch_final_score",
        "quality_delta", "gp_advantage",
    ]].mean()
    print(f"  Mean cost saving          : {global_means['cost_saving_pct']:.1f}%")
    print(f"  Mean trials saved         : {global_means['trials_saved']:+.1f}")
    print(f"  Mean LLM final score      : {global_means['llm_final_score']:.5f}")
    print(f"  Mean GP  final score      : {global_means['gp_final_score']:.5f}")
    print(f"  Mean switching final score: {global_means['switch_final_score']:.5f}")
    print(f"  Mean quality delta (sw-LLM): {global_means['quality_delta']:+.5f}")
    print(f"  Mean GP advantage (GP-LLM) : {global_means['gp_advantage']:+.5f}")
    print(f"{'='*72}\n")

    return summary_df



 
# =============================================================================
#  PAPER TABLE  — CSV-formatted, ready to copy into a spreadsheet / LaTeX
# =============================================================================
 
def print_paper_table(
    summary_df:    pd.DataFrame,
    output_dir:    str   = "results",
    threshold_pct: float = 0.95,
):
    """
    Aggregate summary_df to one row per (model, dataset) — mean ± std across
    all feasible (seed, start) runs — and print it in two CSV formats:
 
      TABLE 1 — Cost & efficiency
        Model, Dataset, tau (mean±std), Cost_LLM ($), Cost_Switch ($),
        Cost_Saving (%), Trials_to_Target_LLM, Trials_to_Target_Switch,
        Trials_Saved
 
      TABLE 2 — Score quality
        Model, Dataset, Score_LLM, Score_GP, Score_Switch,
        Quality_Delta (Switch-LLM), GP_Advantage (GP-LLM),
        LLM_Plateaued (%)
 
    Both tables are printed to the terminal AND saved as CSV files so you
    can paste directly into Excel / Overleaf.
    """
 
    # feas = summary_df[summary_df["feasible"] == True].copy()
    feas = summary_df[summary_df["feasible"] == True].copy() if "feasible" in summary_df.columns else summary_df.copy()
    if feas.empty:
        print("[paper_table] No feasible runs — nothing to print.")
        return
 
    # ── Aggregate: mean and std per (model, dataset) ──────────────────────────
    num_cols = [
        "tau",
        "cost_pure_llm_usd", "cost_switching_usd", "cost_saving_pct",
        "llm_trials_to_target", "switch_trials_to_target", "trials_saved",
        "llm_final_score", "gp_final_score", "switch_final_score",
        "quality_delta", "gp_advantage", "llm_improvement_after_tau",
    ]
    grp   = feas.groupby(["model", "dataset"])
    means = grp[num_cols].mean()
    stds  = grp[num_cols].std().fillna(0)
    n_runs = grp["tau"].count().rename("n_runs")
    plateaued_pct = grp["llm_plateaued"].mean().mul(100).rename("llm_plateaued_pct")
 
    agg = means.join(stds, lsuffix="_mean", rsuffix="_std") \
               .join(n_runs).join(plateaued_pct).reset_index()
 
    def ms(col):
        """Format 'mean ± std' string."""
        return agg.apply(
            lambda r: f"{r[col+'_mean']:.4f}±{r[col+'_std']:.4f}", axis=1)
 
    def ms2(col, fmt=".1f"):
        return agg.apply(
            lambda r: f"{r[col+'_mean']:{fmt}}±{r[col+'_std']:{fmt}}", axis=1)
 
    # ── TABLE 1: Cost & Efficiency ────────────────────────────────────────────
    t1 = pd.DataFrame({
        "Model":                   agg["model"].str.upper(),
        "Dataset":                 agg["dataset"],
        "N_runs":                  agg["n_runs"].astype(int),
        "tau (mean±std)":          ms2("tau", ".1f"),
        "Cost_PureLLM ($)":        agg["cost_pure_llm_usd_mean"].map("{:.6f}".format),
        "Cost_Switching ($)":      agg["cost_switching_usd_mean"].map("{:.6f}".format),
        "Cost_Saving (%)":         ms2("cost_saving_pct", ".1f"),
        "Trials_LLM":              ms2("llm_trials_to_target", ".1f"),
        "Trials_Switch":           ms2("switch_trials_to_target", ".1f"),
        "Trials_Saved":            agg["trials_saved_mean"].map("{:+.1f}".format),
    })
 
    # ── TABLE 2: Score Quality ────────────────────────────────────────────────
    t2 = pd.DataFrame({
        "Model":                   agg["model"].str.upper(),
        "Dataset":                 agg["dataset"],
        "Score_LLM (mean±std)":    ms("llm_final_score"),
        "Score_GP  (mean±std)":    ms("gp_final_score"),
        "Score_Switch (mean±std)": ms("switch_final_score"),
        "Quality_Delta(Sw-LLM)":   agg["quality_delta_mean"].map("{:+.4f}".format),
        "GP_Advantage(GP-LLM)":    agg["gp_advantage_mean"].map("{:+.4f}".format),
        "LLM_Improvement>tau":     agg["llm_improvement_after_tau_mean"].map("{:.4f}".format),
        "LLM_Plateaued (%)":       agg["llm_plateaued_pct"].map("{:.0f}%".format),
    })
 
    # ── Global summary row ────────────────────────────────────────────────────
    def global_ms(col, fmt=".4f"):
        m = feas[col].mean(); s = feas[col].std()
        return f"{m:{fmt}}±{s:{fmt}}"
 
    global_t1 = pd.DataFrame([{
        "Model":              "ALL",
        "Dataset":            "ALL",
        "N_runs":             len(feas),
        "tau (mean±std)":     global_ms("tau", ".1f"),
        "Cost_PureLLM ($)":   f"{feas['cost_pure_llm_usd'].mean():.6f}",
        "Cost_Switching ($)": f"{feas['cost_switching_usd'].mean():.6f}",
        "Cost_Saving (%)":    global_ms("cost_saving_pct", ".1f"),
        "Trials_LLM":         global_ms("llm_trials_to_target", ".1f"),
        "Trials_Switch":      global_ms("switch_trials_to_target", ".1f"),
        "Trials_Saved":       f"{feas['trials_saved'].mean():+.1f}",
    }])
    global_t2 = pd.DataFrame([{
        "Model":                   "ALL",
        "Dataset":                 "ALL",
        "Score_LLM (mean±std)":    global_ms("llm_final_score"),
        "Score_GP  (mean±std)":    global_ms("gp_final_score"),
        "Score_Switch (mean±std)": global_ms("switch_final_score"),
        "Quality_Delta(Sw-LLM)":   f"{feas['quality_delta'].mean():+.4f}",
        "GP_Advantage(GP-LLM)":    f"{feas['gp_advantage'].mean():+.4f}",
        "LLM_Improvement>tau":     f"{feas['llm_improvement_after_tau'].mean():.4f}",
        "LLM_Plateaued (%)":       f"{feas['llm_plateaued'].mean()*100:.0f}%",
    }])
 
    t1_full = pd.concat([t1, global_t1], ignore_index=True)
    t2_full = pd.concat([t2, global_t2], ignore_index=True)
 
    # ── Print to terminal ─────────────────────────────────────────────────────
    sep = "=" * 80
    thr_str = f"{threshold_pct*100:.0f}%"
 
    print(f"\n{sep}")
    print(f"  PAPER TABLE 1 — Cost & Efficiency  (threshold={thr_str})")
    print(f"  Copy the CSV block below directly into Excel or your LaTeX table")
    print(sep)
    print(t1_full.to_csv(index=False))
 
    print(f"{sep}")
    print(f"  PAPER TABLE 2 — Score Quality  (threshold={thr_str})")
    print(f"  Quality_Delta > 0 means switching improves over pure LLM")
    print(f"  GP_Advantage  > 0 means GP finds better configs than LLM at T")
    print(sep)
    print(t2_full.to_csv(index=False))
 
    # ── Save to CSV ───────────────────────────────────────────────────────────
    thr_tag = int(threshold_pct * 100)
    p1 = os.path.join(output_dir, f"paper_table1_cost_thr{thr_tag}.csv")
    p2 = os.path.join(output_dir, f"paper_table2_score_thr{thr_tag}.csv")
    t1_full.to_csv(p1, index=False)
    t2_full.to_csv(p2, index=False)
    print(f"[saved paper table 1] {p1}")
    print(f"[saved paper table 2] {p2}\n")



# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation: Pure LLM vs Switching cost/performance comparison."
    )
    parser.add_argument("--output-dir",    default="results",
                        help="Directory with existing CSVs from hybrid_bo_experiment.py")
    parser.add_argument("--model",         default="all",
                        help="svm | dt | rf | mlp | adaboost | all")
    parser.add_argument("--dataset",       default="all",
                        help="iris | wine | digits | breast | diabetes | boston | all")
    parser.add_argument("--threshold",     type=float, default=0.95,
                        help="Fraction of best final score used as performance target (default 0.95)")
    parser.add_argument("--paper-table-only", action="store_true",
                        help="Skip ablation recompute — just reprint paper tables "
                             "from an existing ablation_summary_thr{N}.csv")
    args = parser.parse_args()

    models   = ALL_MODELS   if args.model   == "all" else [args.model]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    if args.paper_table_only:
        # Fast path: just reload the summary CSV and reprint tables
        thr_tag  = int(args.threshold * 100)
        csv_path = os.path.join(args.output_dir,
                                f"ablation_summary_thr{thr_tag}.csv")
        if not os.path.exists(csv_path):
            print(f"[error] {csv_path} not found. "
                  f"Run without --paper-table-only first.")
        else:
            df = pd.read_csv(csv_path)
            print_paper_table(df, args.output_dir, args.threshold)
    else:
        run_ablation(
            output_dir    = args.output_dir,
            models        = models,
            datasets      = datasets,
            threshold_pct = args.threshold,
        )