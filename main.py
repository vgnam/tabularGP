"""
TabularGP - Config-Driven Main Script
Loads dataset from OpenML, runs regression or classification based on config.yaml
"""

import yaml
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from fastai.tabular.all import *
from tabularGP import tabularGP_learner
from tabularGP.kernel import ProductOfSumsKernel, WeightedSumKernel, WeightedProductKernel, NeuralKernel
from tabularGP.prior import ConstantPrior, ZeroPrior, LinearPrior
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Không sử dụng GPU
# =============================================================================
# Load Config
# =============================================================================
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# =============================================================================
# Kernel / Prior lookup
# =============================================================================
KERNELS = {
    "ProductOfSumsKernel": ProductOfSumsKernel,
    "WeightedSumKernel": WeightedSumKernel,
    "WeightedProductKernel": WeightedProductKernel,
    "NeuralKernel": NeuralKernel,
}

PRIORS = {
    "ConstantPrior": ConstantPrior,
    "ZeroPrior": ZeroPrior,
    "LinearPrior": LinearPrior,
}

# =============================================================================
# Load OpenML Dataset
# =============================================================================
def load_openml_dataset(dataset_id, target=None, n_samples=None):
    """Fetch dataset from OpenML and return a pandas DataFrame + target column name."""
    print(f"Fetching OpenML dataset id={dataset_id}...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    df = data.frame

    # Determine target column
    if target is None:
        target = data.target_names[0] if isinstance(data.target_names, list) else data.target_names
    print(f"Target column: '{target}'")

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # Subsample if requested
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    print(f"Dataset: {data.details.get('name', 'unknown')} | Shape: {df.shape}")
    return df, target

# =============================================================================
# Detect column types
# =============================================================================
def detect_columns(df, target):
    """Auto-detect categorical and continuous columns."""
    cat_names = []
    cont_names = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == "object" or df[col].dtype.name == "category" or df[col].nunique() < 20:
            cat_names.append(col)
        else:
            cont_names.append(col)
    return cat_names, cont_names

# =============================================================================
# Main
# =============================================================================
def main():
    cfg = load_config()

    problem = cfg["problem"]
    dataset_id = cfg["dataset_id"]
    target = cfg.get("target")
    n_samples = cfg.get("n_samples")
    nb_training_points = cfg.get("nb_training_points", 500)
    epochs = cfg.get("epochs", 5)
    lr = cfg.get("lr", 1e-3)
    kernel_name = cfg.get("kernel", "ProductOfSumsKernel")
    prior_name = cfg.get("prior", "ConstantPrior")
    noise = cfg.get("noise", 0.01)
    plot_fi = cfg.get("plot_feature_importance", False)

    kernel = KERNELS[kernel_name]
    prior = PRIORS[prior_name]

    print("=" * 60)
    print(f"TabularGP | Problem: {problem} | Kernel: {kernel_name} | Prior: {prior_name}")
    print("=" * 60)

    # --- Load data ---
    df, target = load_openml_dataset(dataset_id, target, n_samples)
    cat_names, cont_names = detect_columns(df, target)
    print(f"Categorical features ({len(cat_names)}): {cat_names}")
    print(f"Continuous features  ({len(cont_names)}): {cont_names}")

    # --- Convert categorical columns to string (fastai Categorify needs this) ---
    for col in cat_names:
        df[col] = df[col].astype(str)

    # --- Also handle target for classification ---
    if problem == "classification":
        df[target] = df[target].astype(str)

    # --- Build DataLoaders ---
    procs = [Categorify, FillMissing, Normalize]

    if problem == "classification":
        y_block = CategoryBlock()
    else:
        y_block = None  # default: RegressionBlock

    splits = RandomSplitter(valid_pct=0.2, seed=42)(range(len(df)))

    to = TabularPandas(
        df, procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=target,
        y_block=y_block,
        splits=splits,
    )
    data = to.dataloaders(bs=64)

    # --- Build and Train ---
    if problem == "classification":
        metric = accuracy
    else:
        metric = rmse

    learn = tabularGP_learner(
        data,
        nb_training_points=nb_training_points,
        kernel=kernel,
        prior=prior,
        noise=noise,
        metrics=metric,
    )

    print(f"\nTraining for {epochs} epochs with lr={lr}...")
    learn.fit_one_cycle(epochs, lr_max=lr)

    # --- Sample Prediction ---
    print("\n" + "-" * 40)
    print("Sample predictions (first 5 validation rows):")
    print("-" * 40)
    preds, targets = learn.get_preds(dl=data[1])
    val_idx = splits[1][:5]
    for i in range(min(5, len(preds))):
        pred = preds[i]
        actual = df.iloc[val_idx[i]][target] if i < len(val_idx) else "N/A"
        if problem == "classification":
            pred_label = pred.argmax().item()
            print(f"  [{i+1}] Predicted: class {pred_label} | Actual: {actual}")
        else:
            print(f"  [{i+1}] Predicted: {pred.item():.4f} | Actual: {actual}")

    # --- Feature Importance ---
    print("\n" + "-" * 40)
    print("Feature Importance:")
    print("-" * 40)
    importance = learn.feature_importance.sort_values('Importance', ascending=False)
    print(importance.to_string(index=False))

    if plot_fi:
        learn.plot_feature_importance()
        import matplotlib.pyplot as plt
        plt.savefig("feature_importance.png", bbox_inches="tight", dpi=150)
        print("\nPlot saved to feature_importance.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
