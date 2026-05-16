import argparse
from src.hybrid_bo.core import run_single
from utils.io import save_csv
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="svm")
    parser.add_argument("--dataset", default="wine")
    parser.add_argument("--trials",  type=int, default=10)
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--engine",  default="gpt-3.5-turbo")
    args = parser.parse_args()

    records, llm_log, switch_result = run_single(
        args.model, args.dataset, args.trials,
        args.seed, args.start, args.engine)

    save_csv(pd.DataFrame(records), f"results/{args.model}_{args.dataset}_local.csv")
    print(f"Done. Switch point: tau={switch_result.tau}")

if __name__ == "__main__":
    main()