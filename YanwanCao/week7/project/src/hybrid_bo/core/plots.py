import matplotlib
matplotlib.use("Agg")
import pandas as pd
import os
import numpy as np

from src.hybrid_bo.data.spaces import MODEL_SPACES
from src.hybrid_bo.core.metrics import DISTANCE_METRICS


def _pick_best_seed(df: pd.DataFrame) -> int:
    t_max = df["trial"].max()
    final = df[df["trial"] == t_max]
    seed_quality = (
        final.groupby("seed")
        .apply(lambda g: (g["best_llm"].max() + g["best_gp"].max()) / 2)
    )
    best = int(seed_quality.idxmax())
    print(f"  [best-seed selection] scores per seed:\n{seed_quality.to_string()}")
    print(f"  → selected seed={best} "
          f"(mean final score = {seed_quality[best]:.5f})")
    return best


def plot_all_starts(df: pd.DataFrame, model_name: str,
                    dataset: str, output_dir: str = "results"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return

    seeds     = sorted(df["seed"].unique())
    start_ids = sorted(df["start_idx"].unique())
    colors    = cm.tab10(np.linspace(0, 0.9, max(len(start_ids), 1)))
    tag       = f"{model_name}_{dataset}"
    n_dist    = len(DISTANCE_METRICS)
    ncols     = n_dist + 1

    for seed in seeds:
        seed_df = df[df["seed"] == seed]
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
        fig.suptitle(
            f"{model_name.upper()} / {dataset} / seed={seed}  --  "
            f"all {len(start_ids)} starts",
            fontsize=10)

        for i, sid in enumerate(start_ids):
            s_df   = seed_df[seed_df["start_idx"] == sid].sort_values("trial")
            trials = s_df["trial"].values
            c      = colors[i]
            for j, (key, _, long_name, ylabel) in enumerate(DISTANCE_METRICS):
                axes[j].plot(trials, s_df[key], color=c, lw=1.5,
                             marker="o", ms=2, label=f"start {sid}")
            axes[-1].plot(trials, s_df["best_llm"], color=c, lw=1.5,
                          label=f"LLM s{sid}")
            axes[-1].plot(trials, s_df["best_gp"],  color=c, lw=1.5,
                          ls="--", label=f"GP s{sid}")

        for j, (key, _, long_name, ylabel) in enumerate(DISTANCE_METRICS):
            axes[j].set(title=long_name, xlabel="Trial t", ylabel=ylabel)
            axes[j].set_ylim(bottom=0)
            axes[j].legend(fontsize=6, ncol=2)
        axes[-1].set(title="Best score", xlabel="Trial t", ylabel="Best CV score")
        axes[-1].set_ylim(bottom=0)
        axes[-1].legend(fontsize=6, ncol=2)

        plt.tight_layout()
        out = os.path.join(output_dir, f"{tag}_seed{seed}_allstarts.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved plot] {out}")
        plt.close()

    trials = sorted(df["trial"].unique())
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(
        f"{model_name.upper()} / {dataset}  --  "
        f"mean+/-std across {len(seeds)} seeds, all {len(start_ids)} starts",
        fontsize=10)

    for i, sid in enumerate(start_ids):
        s_df = df[df["start_idx"] == sid].groupby("trial")
        c    = colors[i]
        for j, (key, _, long_name, ylabel) in enumerate(DISTANCE_METRICS):
            mu  = s_df[key].mean().values
            std = s_df[key].std().fillna(0).values
            axes[j].plot(trials, mu, color=c, lw=1.8, label=f"start {sid}")
            axes[j].fill_between(trials, mu - std, mu + std, alpha=0.15, color=c)
            axes[j].set(title=f"{long_name} (mean+/-std)",
                         xlabel="Trial t", ylabel=ylabel)
            axes[j].set_ylim(bottom=0)

        mu_llm  = s_df["best_llm"].mean().values
        mu_gp   = s_df["best_gp"].mean().values
        std_llm = s_df["best_llm"].std().fillna(0).values
        std_gp  = s_df["best_gp"].std().fillna(0).values
        axes[-1].plot(trials, mu_llm, color=c, lw=1.8, label=f"LLM s{sid}")
        axes[-1].plot(trials, mu_gp,  color=c, lw=1.8, ls="--", label=f"GP s{sid}")
        axes[-1].fill_between(trials, mu_llm - std_llm, mu_llm + std_llm,
                               alpha=0.12, color=c)
        axes[-1].fill_between(trials, mu_gp - std_gp, mu_gp + std_gp,
                               alpha=0.12, color=c)

    axes[-1].set(title="Best score (mean+/-std)", xlabel="Trial t",
                 ylabel="Best CV score")
    axes[-1].set_ylim(bottom=0)
    for ax in axes:
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    out = os.path.join(output_dir, f"{tag}_allstarts_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved plot] {out}")
    plt.close()

    hp_keys = list(MODEL_SPACES[model_name].keys())
    n_hp    = len(hp_keys)
    fig, axes = plt.subplots(1, n_hp, figsize=(5 * n_hp, 4))
    if n_hp == 1:
        axes = [axes]
    fig.suptitle(
        f"{model_name.upper()} / {dataset} -- proposed HP values\n"
        f"(seed=0; solid=LLM, dashed=GP; each colour=one start)",
        fontsize=10)

    seed0_df = df[df["seed"] == sorted(df["seed"].unique())[0]]
    for ax, hp in zip(axes, hp_keys):
        for i, sid in enumerate(start_ids):
            s_df = seed0_df[seed0_df["start_idx"] == sid].sort_values("trial")
            ax.plot(s_df["trial"], s_df[f"llm_{hp}"],
                    color=colors[i], lw=1.4, label=f"LLM s{sid}")
            ax.plot(s_df["trial"], s_df[f"gp_{hp}"],
                    color=colors[i], lw=1.4, ls="--", label=f"GP s{sid}")
        ax.set(title=hp, xlabel="Trial t", ylabel="Proposed value")
        ax.legend(fontsize=6, ncol=2)
        if MODEL_SPACES[model_name][hp][0] == "log":
            ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(output_dir, f"{tag}_hp_trajectories.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved plot] {out}")
    plt.close()


def plot_best_seed_distance_overlay(df: pd.DataFrame, model_name: str,
                                     dataset: str, output_dir: str = "results"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tag        = f"{model_name}_{dataset}"
    best_seed  = _pick_best_seed(df)
    seed_df    = df[df["seed"] == best_seed]

    t_max      = df["trial"].max()
    final_df   = seed_df[seed_df["trial"] == t_max]
    best_start = int(final_df.loc[final_df["best_llm"].idxmax(), "start_idx"])
    best_df    = seed_df[seed_df["start_idx"] == best_start].sort_values("trial")
    trials     = best_df["trial"].values
    start_ids  = sorted(seed_df["start_idx"].unique())

    dist_colors  = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
    dist_styles  = ["-",        "--",      "-.",      ":",       (0,(3,1,1,1))]
    dist_markers = ["o",        "s",       "^",       "D",       "P"]

    fig, (ax_dist, ax_score) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_name.upper()} / {dataset}  --  "
        f"best seed={best_seed}, best start={best_start}",
        fontsize=10,
    )

    for (key, _, long_name, ylabel), color, ls, marker in zip(
        DISTANCE_METRICS, dist_colors, dist_styles, dist_markers
    ):
        vals = best_df[key].values
        ax_dist.plot(trials, vals, color=color, lw=2.0, ls=ls, marker=marker,
                     ms=4, markevery=max(1, len(trials) // 10),
                     label=f"{long_name} ({key})")

    ax_dist.set(
        title="All 5 distance metrics — LLM vs GP trajectory",
        xlabel="Trial t", ylabel="Distance (warped HP space)",
    )
    ax_dist.set_ylim(bottom=0)
    ax_dist.legend(fontsize=8, loc="upper right")
    ax_dist.grid(True, alpha=0.3)

    for (key, _, _, _), color in zip(DISTANCE_METRICS, dist_colors):
        fv = best_df[key].iloc[-1]
        ax_dist.annotate(f"{fv:.3f}", xy=(trials[-1], fv),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=7, color=color, va="center")

    score_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(start_ids), 1)))
    for i, sid in enumerate(start_ids):
        s_df = seed_df[seed_df["start_idx"] == sid].sort_values("trial")
        lw   = 2.5 if sid == best_start else 1.2
        al   = 1.0 if sid == best_start else 0.5
        star = " [best]" if sid == best_start else ""
        ax_score.plot(s_df["trial"], s_df["best_llm"], color=score_colors[i],
                      lw=lw, alpha=al, label=f"LLM start={sid}{star}")
        ax_score.plot(s_df["trial"], s_df["best_gp"],  color=score_colors[i],
                      lw=lw, alpha=al, ls="--", label=f"GP  start={sid}{star}")

    ax_score.set(
        title=f"Best score per trial (seed={best_seed}, all starts)",
        xlabel="Trial t", ylabel="Best CV score (running max)",
    )
    ax_score.set_ylim(bottom=0)
    ax_score.legend(fontsize=7, ncol=2)
    ax_score.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, f"{tag}_best_seed_overlay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved best-seed overlay]      {out}")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"{model_name.upper()} / {dataset}  --  Normalised distance trends\n"
        f"(seed={best_seed}, start={best_start})", fontsize=10)
    for (key, _, long_name, _), color, ls, marker in zip(
        DISTANCE_METRICS, dist_colors, dist_styles, dist_markers
    ):
        vals = best_df[key].values
        norm = vals / (vals.max() + 1e-12)
        ax.plot(trials, norm, color=color, lw=2.0, ls=ls, marker=marker,
                ms=4, markevery=max(1, len(trials) // 10),
                label=f"{long_name} ({key})")

    ax.set(title="All 5 distances normalised to [0,1]",
           xlabel="Trial t", ylabel="Normalised distance")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_norm = os.path.join(output_dir, f"{tag}_best_seed_overlay_normalised.png")
    fig.savefig(out_norm, dpi=150, bbox_inches="tight")
    print(f"[saved normalised overlay]     {out_norm}")
    plt.close()