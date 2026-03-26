"""
Benchmark: TabularGP vs Random Forest
Demonstrates datasets where TabularGP outperforms Random Forest.

GP excels on:
  - Noisy datasets (GP explicitly models noise, RF overfits to it)
  - Small datasets (< 1000 points)
  - Smooth underlying functions buried in noise

We test on:
  1. Synthetic noisy function - 200 samples, high noise
  2. OpenML: Servo (id=188) - 167 rows, noisy physical measurements
"""

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Tee stdout to file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_file = open("benchmark_results.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
from fastai.tabular.all import *
from tabularGP import tabularGP_learner
from tabularGP.kernel import WeightedSumKernel
from tabularGP.prior import LinearPrior


def run_tabularGP(df, target, cat_names, cont_names, train_idx, val_idx,
                  epochs=15, lr=1e-3, noise=0.1):
    """Train TabularGP and return RMSE on validation set."""
    df_copy = df.copy()
    for col in cat_names:
        df_copy[col] = df_copy[col].astype(str)

    procs = [Categorify, FillMissing, Normalize]
    splits = (list(train_idx), list(val_idx))

    to = TabularPandas(
        df_copy, procs=procs,
        cat_names=cat_names, cont_names=cont_names,
        y_names=target, splits=splits,
    )
    data = to.dataloaders(bs=64)

    nb_pts = min(len(train_idx), 500)

    learn = tabularGP_learner(
        data,
        nb_training_points=nb_pts,
        kernel=WeightedSumKernel,
        prior=LinearPrior,
        noise=noise,
        metrics=rmse,
    )

    with learn.no_bar(), learn.no_logging():
        learn.fit_one_cycle(epochs, lr_max=lr)

    preds, targets = learn.get_preds(dl=data[1])
    return np.sqrt(mean_squared_error(targets.numpy(), preds.numpy()))


def run_random_forest(df, target, train_idx, val_idx):
    """Train Random Forest and return RMSE on validation set."""
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(0)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train.astype(float))
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val.astype(float), preds))


# =========================================================================
# Dataset 1: Synthetic NOISY function
# GP models noise explicitly; RF overfits to noise
# =========================================================================
def make_noisy_dataset(n=200, noise_std=1.5, seed=42):
    """
    Simple smooth function + HIGH noise.
    True signal: y = sin(x1*pi) + x2^2 - x3  (range ~[-3, 3])
    Noise std = 1.5 (signal-to-noise ratio is low)
    
    GP handles this well because it has a noise parameter.
    RF tries to fit every noisy point -> overfits.
    """
    np.random.seed(seed)
    X = np.random.uniform(-2, 2, (n, 4))
    signal = np.sin(X[:, 0] * np.pi) + X[:, 1]**2 - X[:, 2]
    noise = np.random.normal(0, noise_std, n)
    y = signal + noise

    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["target"] = y
    return df, "target"


# =========================================================================
# Dataset 2: OpenML Servo (id=188, 167 rows, noisy physical measurements)
# =========================================================================
def load_servo_dataset():
    data = fetch_openml(data_id=188, as_frame=True, parser="auto")
    df = data.frame
    target = data.target_names[0] if isinstance(data.target_names, list) else data.target_names
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    print(f"  Servo dataset: {df.shape}, target='{target}'")
    return df, target


def detect_columns(df, target):
    cat_names, cont_names = [], []
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object" or df[col].dtype.name == "category" or df[col].nunique() < 20:
            cat_names.append(col)
        else:
            cont_names.append(col)
    return cat_names, cont_names


def benchmark_dataset(name, df, target, epochs=15, lr=1e-3, noise=0.1, n_runs=3):
    """Run multiple random splits and average results."""
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    cat_names, cont_names = detect_columns(df, target)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Samples: {len(df)} | Cat: {len(cat_names)}, Cont: {len(cont_names)}")
    print(f"  Running {n_runs} splits...")
    print(f"{'='*60}")

    gp_scores, rf_scores = [], []

    for run in range(n_runs):
        seed = 42 + run
        np.random.seed(seed)
        indices = np.random.permutation(len(df))
        split = int(0.8 * len(df))
        train_idx, val_idx = indices[:split], indices[split:]

        rf_rmse = run_random_forest(df, target, train_idx, val_idx)
        rf_scores.append(rf_rmse)

        try:
            gp_rmse = run_tabularGP(
                df, target, cat_names, cont_names,
                train_idx, val_idx,
                epochs=epochs, lr=lr, noise=noise,
            )
            gp_scores.append(gp_rmse)
        except Exception as e:
            print(f"  [Run {run+1}] GP error: {e}")
            gp_scores.append(float('inf'))

        print(f"  [Run {run+1}] GP={gp_scores[-1]:.4f}  RF={rf_scores[-1]:.4f}")

    gp_mean, rf_mean = np.mean(gp_scores), np.mean(rf_scores)
    winner = "TabularGP" if gp_mean < rf_mean else "RF"
    diff = (rf_mean - gp_mean) / rf_mean * 100

    print(f"\n  GP avg:  {gp_mean:.4f} ± {np.std(gp_scores):.4f}")
    print(f"  RF avg:  {rf_mean:.4f} ± {np.std(rf_scores):.4f}")
    print(f"  Winner: {winner} ({abs(diff):.1f}%)")
    return {"dataset": name, "gp_rmse": gp_mean, "rf_rmse": rf_mean, "winner": winner}


def main():
    print("=" * 60)
    print("  TabularGP vs Random Forest (Noisy Datasets)")
    print("  GP models noise explicitly -> avoids overfitting")
    print("=" * 60)

    results = []

    # Noisy synthetic (high noise, GP's noise param helps)
    df1, t1 = make_noisy_dataset(n=200, noise_std=1.5)
    results.append(benchmark_dataset(
        "Synthetic NOISY (n=200, noise_std=1.5)", df1, t1,
        epochs=15, lr=1e-3, noise=0.3,
    ))

    # Servo
    try:
        df2, t2 = load_servo_dataset()
        results.append(benchmark_dataset(
            "Servo (OpenML #188, n=167)", df2, t2,
            epochs=15, lr=1e-3, noise=0.1,
        ))
    except Exception as e:
        print(f"Servo failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
