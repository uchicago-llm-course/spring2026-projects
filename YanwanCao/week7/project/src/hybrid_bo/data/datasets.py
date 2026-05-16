from sklearn import datasets as sk_datasets


def _load_boston():
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name="boston", version=1, as_frame=False, parser="auto")
        return ds.data, ds.target.astype(float)
    except Exception:
        ds = sk_datasets.fetch_california_housing()
        return ds.data, ds.target
    
    
def load_dataset(name: str):
    entry, task = DATASET_REGISTRY[name]
    if entry == "boston":
        X, y = _load_boston()
    else:
        data = entry()
        X, y = data.data, data.target
    return X, y, task

DATASET_REGISTRY: dict = {
    "iris":     (sk_datasets.load_iris,          "clf"),
    "wine":     (sk_datasets.load_wine,          "clf"),
    "digits":   (sk_datasets.load_digits,        "clf"),
    "breast":   (sk_datasets.load_breast_cancer, "clf"),
    "diabetes": (sk_datasets.load_diabetes,      "reg"),
    "boston":   ("boston",                        "reg"),
}

ALL_DATASETS = list(DATASET_REGISTRY.keys())