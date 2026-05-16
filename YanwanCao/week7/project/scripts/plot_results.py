import matplotlib
matplotlib.use("Agg")
import pandas as pd
import argparse
from src.hybrid_bo.core.plots import plot_all_starts, plot_best_seed_distance_overlay

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--outdir", default="results")
args = parser.parse_args()

df = pd.read_csv(args.csv)
plot_all_starts(df, args.model, args.dataset, args.outdir)
plot_best_seed_distance_overlay(df, args.model, args.dataset, args.outdir)