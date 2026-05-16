# utils/logging.py
def trial_header(model, dataset, seed, start_idx, task):
    sep = "-" * 82
    print(f"\n  {sep}")
    print(f"  MODEL={model.upper()}  DATASET={dataset}  "
          f"seed={seed}  start={start_idx}  task={task}")
    print(f"  {sep}")

def trial_row(t, lc, gc, ls, gs, hp_keys, dists, is_new_best):
    print(f"\n  [t={t:03d}] LLM-> " +
          " | ".join(f"{k}={lc[k]:.4g}" for k in hp_keys) +
          f"  score={ls:.5f}{'  ** NEW BEST **' if is_new_best else ''}")
    print(f"         GP  -> " +
          " | ".join(f"{k}={gc[k]:.4g}" for k in hp_keys) +
          f"  score={gs:.5f}")
    print("         dists: " +
          "  ".join(f"{key}={v:.4f}" for key, v in dists.items()))