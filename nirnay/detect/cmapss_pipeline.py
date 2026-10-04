"""
C-MAPSS Data Pipeline — Load, preprocess, and split.

Status: [REAL]
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

# Column names for C-MAPSS datasets
CMAPSS_COLS = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]


def load_dataset(data_dir: str, subset: str = "FD001"
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load a C-MAPSS subset (FD001–FD004).

    Returns (train_df, test_df, rul_true).
    """
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=CMAPSS_COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=CMAPSS_COLS)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"]).squeeze()

    return train_df, test_df, rul_true


def cluster_regimes(df: pd.DataFrame, n_clusters: int = 6,
                    seed: int = 42) -> pd.DataFrame:
    """
    Cluster operating settings into discrete regimes using KMeans.
    Adds a 'regime' column.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    df = df.copy()
    df["regime"] = km.fit_predict(df[OP_COLS].values)
    return df


def normalise_per_regime(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
                         ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    """
    Per-regime z-score normalisation.  Statistics computed on training data only.
    Returns (norm_train, norm_test, stats_dict).
    """
    stats = {}
    train_out = train_df.copy()

    for regime in train_df["regime"].unique():
        mask = train_df["regime"] == regime
        subset = train_df.loc[mask, SENSOR_COLS]
        mu = subset.mean()
        sigma = subset.std().replace(0, 1)
        stats[regime] = {"mean": mu, "std": sigma}
        train_out.loc[mask, SENSOR_COLS] = (subset - mu) / sigma

    test_out = None
    if test_df is not None:
        test_out = test_df.copy()
        for regime in test_df["regime"].unique():
            mask = test_df["regime"] == regime
            if regime in stats:
                mu = stats[regime]["mean"]
                sigma = stats[regime]["std"]
                test_out.loc[mask, SENSOR_COLS] = (test_df.loc[mask, SENSOR_COLS] - mu) / sigma

    return train_out, test_out, stats


def add_rul_target(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Add piecewise-linear RUL target.
    For training data: RUL = max_life - cycle, capped at max_rul.
    """
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["rul"] = df.apply(lambda r: max_cycles[r["engine_id"]] - r["cycle"], axis=1)
    df["rul"] = df["rul"].clip(upper=max_rul)
    return df


def make_windows(df: pd.DataFrame, window_size: int = 30,
                 features: list = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for ML models.
    Returns (X, y) where X is (n_samples, window_size, n_features) and y is RUL at the end.
    """
    if features is None:
        features = SENSOR_COLS

    X_list, y_list = [], []

    for eid in df["engine_id"].unique():
        engine = df[df["engine_id"] == eid].sort_values("cycle")
        vals = engine[features].values
        ruls = engine["rul"].values

        for i in range(len(vals) - window_size + 1):
            X_list.append(vals[i:i + window_size])
            y_list.append(ruls[i + window_size - 1])

    return np.array(X_list), np.array(y_list)
