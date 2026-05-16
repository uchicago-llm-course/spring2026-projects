import matplotlib
matplotlib.use("Agg")
import modal
import pandas as pd
import os

from config import VOLUME_PATH, SWITCH_EPSILON_FRACTION, LLM_INPUT_PRICE_PER_MILLION, LLM_OUTPUT_PRICE_PER_MILLION
from src.hybrid_bo.core import run_single, plot_switch_results, plot_all_starts, plot_best_seed_distance_overlay, summarise_switch_results
from src.hybrid_bo.data  import ALL_DATASETS, ALL_MODELS, MODEL_SPACES, DISTANCE_METRICS, DIST_KEYS


app = modal.App("hybrid-bo-experiment-v4")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn>=1.4",
        "scipy>=1.12",
        "POT>=0.9",
        "numpy>=1.26",
        "pandas>=2.2",
        "matplotlib>=3.8",
        "openai>=1.20",
        "gurobipy",
    )
    .add_local_dir("config", remote_path="/root/config")   # ← add this
    .add_local_dir("src",    remote_path="/root/src")      # ← add this
)
openai_secret = modal.Secret.from_name("openai-secret")
volume        = modal.Volume.from_name("hybrid-bo-results-v4", create_if_missing=True)


def _print_column_guide(df: pd.DataFrame, model_name: str, dataset: str):
    hp_keys = list(MODEL_SPACES[model_name].keys())
    print("\n  CSV column guide:")
    print("  +-- identifiers -------------------------------------------")
    for c in ["model", "dataset", "trial", "seed", "start_idx"]:
        print(f"  |  {c}")
    print("  +-- distance metrics (LLM traj vs GP traj) ----------------")
    for key, _, long, ylabel in DISTANCE_METRICS:
        print(f"  |  {key:<12} -- {long}")
    print("  +-- scores ------------------------------------------------")
    print("  |  gp_score_t   -- GP raw score at trial t")
    print("  |  llm_score_t  -- LLM raw score at trial t")
    print("  |  best_gp      -- running max GP score up to trial t")
    print("  |  best_llm     -- running max LLM score up to trial t")
    print("  +-- GP proposed HPs ----------------------------------------")
    for k in hp_keys:
        print(f"  |  gp_{k}")
    print("  +-- LLM proposed HPs ---------------------------------------")
    for k in hp_keys:
        print(f"  |  llm_{k}")
    print("  +-----------------------------------------------------------\n")


@app.function(image=image, secrets=[openai_secret],
              volumes={VOLUME_PATH: volume}, cpu=8, timeout=7200)
def run_single_modal(model_name, dataset, n_trials, seed, start_idx, engine):
    return run_single(model_name, dataset, n_trials, seed, start_idx, engine)

@app.local_entrypoint()
def main(
    dataset:    str = "all",
    model:      str = "all",
    num_trials: int = 25,
    num_seeds:  int = 3,
    num_starts: int = 3,
    engine:     str = "gpt-3.5-turbo",
    output_dir: str = "results",
):
    datasets = ALL_DATASETS if dataset == "all" else [dataset]
    models   = ALL_MODELS   if model   == "all" else [model]

    total_containers = len(datasets) * len(models) * num_seeds * num_starts
    print("\n" + "=" * 68)
    print("  HYBRID BO EXPERIMENT v4 — CONFIG")
    print("=" * 68)
    print(f"  datasets      : {datasets}")
    print(f"  models        : {models}")
    print(f"  num_seeds     : {num_seeds}")
    print(f"  num_starts    : {num_starts}")
    print(f"  num_trials    : {num_trials}")
    print(f"  switch epsilon fraction: {SWITCH_EPSILON_FRACTION}")
    print(f"  LLM pricing   : ${LLM_INPUT_PRICE_PER_MILLION}/1M input, "
          f"${LLM_OUTPUT_PRICE_PER_MILLION}/1M output")
    print(f"  Total containers: {total_containers}")
    print(f"  engine        : {engine}")
    print(f"  output_dir    : {output_dir}")
    print("=" * 68 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    for ds in datasets:
        for mdl in models:
            print(f"\n{'='*68}")
            print(f"  STARTING: model={mdl.upper()}  dataset={ds}")
            print(f"  Dispatching {num_seeds * num_starts} containers in parallel")
            print(f"{'='*68}")

            all_records:        list = []
            all_llm_logs:       list = []
            all_switch_results: list = []   # ← one SwitchResult per (seed, start)

            all_jobs = [
                (mdl, ds, num_trials, seed, si, engine)
                for seed in range(num_seeds)
                for si   in range(num_starts)
            ]
            results = list(run_single_modal.starmap(all_jobs))

            # ── Unpack 3 values per job (records, llm_log, switch_result) ────
            for records, llm_log, switch_result in results:
                all_records.extend(records)
                all_llm_logs.extend(llm_log)
                all_switch_results.append(switch_result)

            df = pd.DataFrame(all_records)

            traj_path = os.path.join(output_dir, f"{mdl}_{ds}_trajectories.csv")
            df.to_csv(traj_path, index=False)
            print(f"\n[saved trajectories] {traj_path}")
            _print_column_guide(df, mdl, ds)

            llm_df   = pd.DataFrame(all_llm_logs)
            llm_path = os.path.join(output_dir, f"{mdl}_{ds}_llm_proposals.csv")
            llm_df.to_csv(llm_path, index=False)
            print(f"[saved LLM log]      {llm_path}")

            summary_cols = (DIST_KEYS + ["best_gp", "best_llm"] +
                            [f"gp_{k}"  for k in MODEL_SPACES[mdl]] +
                            [f"llm_{k}" for k in MODEL_SPACES[mdl]])
            summary = (df.groupby(["seed", "start_idx", "trial"])
                         [summary_cols].mean())
            sum_path = os.path.join(output_dir, f"{mdl}_{ds}_summary.csv")
            summary.to_csv(sum_path)
            print(f"[saved summary]      {sum_path}")

            print("\n-- Final best per (seed, start) --")
            final = (df[df["trial"] == df["trial"].max()]
                       .groupby(["seed", "start_idx"])
                       [["best_gp", "best_llm"]].max())
            print(final.to_string())

            # ── Existing trajectory plots ─────────────────────────────────────
            plot_all_starts(df, mdl, ds, output_dir)
            plot_best_seed_distance_overlay(df, mdl, ds, output_dir)

            # ── Switching-point summary + plot (new) ──────────────────────────
            df_switch = summarise_switch_results(
                all_switch_results,
                model_name = mdl,
                dataset    = ds,
                output_dir = output_dir,
            )
            plot_switch_results(df_switch, mdl, ds, output_dir, T=num_trials)