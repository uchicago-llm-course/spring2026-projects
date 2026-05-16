from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               AdaBoostClassifier, AdaBoostRegressor)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from src.hybrid_bo.data.datasets import DATASET_REGISTRY, _load_boston, load_dataset


MODEL_SPACES: dict = {
    "svm": {
        "C":                 ("log",    1e-2, 1e5),
        "gamma":             ("log",    1e-6, 1e0),
        "tol":               ("log",    1e-6, 1e-1),
    },
    "dt": {
        "max_depth":         ("linear", 1,    32),
        "min_samples_split": ("linear", 2,    32),
        "min_samples_leaf":  ("linear", 1,    16),
        "ccp_alpha":         ("log",    1e-5, 1e-1),
    },
    "rf": {
        "n_estimators":      ("linear", 10,   500),
        "max_depth":         ("linear", 1,    32),
        "min_samples_split": ("linear", 2,    20),
        "max_features":      ("logit",  0.05, 0.95),
    },
    "mlp": {
        "hidden_layer_size": ("log",    16,   512),
        "alpha":             ("log",    1e-5, 1e-1),
        "learning_rate_init":("log",    1e-4, 1e-1),
        "max_iter":          ("linear", 50,   500),
    },
    "adaboost": {
        "n_estimators":      ("linear", 10,   500),
        "learning_rate":     ("log",    1e-3, 2.0),
        "max_depth":         ("linear", 1,    10),
    },
}

ALL_MODELS = list(MODEL_SPACES.keys())


def build_model(model_name: str, config: dict, task: str, random_state: int = 0):
    if model_name == "svm":
        if task == "clf":
            return SVC(C=config["C"], gamma=config["gamma"],
                       tol=config["tol"], kernel="rbf", random_state=random_state)
        return SVR(C=config["C"], gamma=config["gamma"],
                   tol=config["tol"], kernel="rbf")

    if model_name == "dt":
        kw = dict(max_depth=max(1, int(round(config["max_depth"]))),
                  min_samples_split=max(2, int(round(config["min_samples_split"]))),
                  min_samples_leaf=max(1, int(round(config["min_samples_leaf"]))),
                  ccp_alpha=config["ccp_alpha"], random_state=random_state)
        return (DecisionTreeClassifier(**kw) if task == "clf"
                else DecisionTreeRegressor(**kw))

    if model_name == "rf":
        kw = dict(n_estimators=max(10, int(round(config["n_estimators"]))),
                  max_depth=max(1, int(round(config["max_depth"]))),
                  min_samples_split=max(2, int(round(config["min_samples_split"]))),
                  max_features=float(config["max_features"]),
                  random_state=random_state, n_jobs=-1)
        return (RandomForestClassifier(**kw) if task == "clf"
                else RandomForestRegressor(**kw))

    if model_name == "mlp":
        h = max(16, int(round(config["hidden_layer_size"])))
        kw = dict(hidden_layer_sizes=(h, h), alpha=config["alpha"],
                  learning_rate_init=config["learning_rate_init"],
                  max_iter=max(50, int(round(config["max_iter"]))),
                  random_state=random_state)
        return (MLPClassifier(**kw) if task == "clf"
                else MLPRegressor(**kw))

    if model_name == "adaboost":
        depth  = max(1, int(round(config["max_depth"])))
        base_c = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        base_r = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        kw = dict(n_estimators=max(10, int(round(config["n_estimators"]))),
                  learning_rate=config["learning_rate"], random_state=random_state)
        return (AdaBoostClassifier(estimator=base_c, **kw) if task == "clf"
                else AdaBoostRegressor(estimator=base_r, **kw))

    raise ValueError(f"Unknown model: {model_name}")


def make_objective(model_name: str, dataset_name: str):
    X, y, task = load_dataset(dataset_name)

    def objective(config: dict) -> float:
        mdl     = build_model(model_name, config, task, random_state=0)
        pipe    = Pipeline([("scaler", StandardScaler()), ("mdl", mdl)])
        scoring = "accuracy" if task == "clf" else "neg_mean_squared_error"
        scores  = cross_val_score(pipe, X, y, cv=5, scoring=scoring, n_jobs=-1)
        return float(scores.mean())

    return objective, task