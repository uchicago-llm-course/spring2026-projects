import os
import pandas as pd


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)