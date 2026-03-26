"""
TabularGP - Config-Driven Main Script
Loads dataset from OpenML, runs regression or classification based on config.yaml
"""

import yaml
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.datasets import fetch_openml
from fastai.tabular.all import *
from tabularGP import tabularGP_learner
from tabularGP.kernel import ProductOfSumsKernel, WeightedSumKernel, WeightedProductKernel, NeuralKernel, LLMKernel
from tabularGP.prior import ConstantPrior, ZeroPrior, LinearPrior, LLMPrior
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Không sử dụng GPU
os.environ["NVIDIA_NIM_API_KEY"] = "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"

# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging(cfg):
    """Setup file + console logging. Returns the log file path."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kernel_name = cfg.get("kernel", "ProductOfSumsKernel")
    prior_name = cfg.get("prior", "ConstantPrior")
    dataset_id = cfg.get("dataset_id", "unknown")
    log_file = log_dir / f"run_{timestamp}_ds{dataset_id}_{kernel_name}_{prior_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_file
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
    "LLMKernel": LLMKernel,
}

PRIORS = {
    "ConstantPrior": ConstantPrior,
    "ZeroPrior": ZeroPrior,
    "LinearPrior": LinearPrior,
    "LLMPrior": LLMPrior,
}

# =============================================================================
# Load OpenML Dataset
# =============================================================================
def load_openml_dataset(dataset_id, target=None, n_samples=None):
    """Fetch dataset from OpenML and return a pandas DataFrame + target column name + description."""
    print(f"Fetching OpenML dataset id={dataset_id}...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    df = data.frame

    # Determine target column
    if target is None:
        if isinstance(data.target_names, list) and len(data.target_names) > 0:
            target = data.target_names[0]
        elif isinstance(data.target_names, str) and data.target_names:
            target = data.target_names
        else:
            # Fallback: use the last column
            target = df.columns[-1]
    print(f"Target column: '{target}'")

    # Get dataset description from OpenML
    dataset_name = data.details.get('name', 'unknown')
    dataset_desc = data.get('DESCR', '') or ''
    # Truncate if too long (keep first 500 chars for LLM context)
    if len(dataset_desc) > 500:
        dataset_desc = dataset_desc[:500] + "..."
    description = f"{dataset_name}: {dataset_desc}" if dataset_desc else dataset_name

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # Subsample if requested
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    print(f"Dataset: {dataset_name} | Shape: {df.shape}")
    return df, target, description

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

    # Kernel setup (LLMKernel handled later after data is loaded)
    kernel_extra = None
    if kernel_name == "LLMKernel":
        llm_k_cfg = cfg.get("llm_kernel", {})
        base_kernel_name = llm_k_cfg.get("base_kernel", "WeightedSumKernel")
        kernel_extra = {
            "base_kernel": KERNELS[base_kernel_name],
            "lam": llm_k_cfg.get("lambda", 0.5),
            "trainable_lambda": llm_k_cfg.get("trainable_lambda", True),
        }
        kernel = None  # Will be created after LLM query
    else:
        kernel = KERNELS[kernel_name]

    prior_cls = PRIORS[prior_name]

    # For LLMPrior, we need to create a factory that passes extra config
    if prior_name == "LLMPrior":
        llm_cfg = cfg.get("llm_prior", {})
        llm_lambda = llm_cfg.get("lambda", 0.5)
        llm_description = llm_cfg.get("dataset_description", "")
        llm_models = llm_cfg.get("models", [])
        llm_trainable = llm_cfg.get("trainable_lambda", True)
        # Will be set after we know feature names and problem type
        prior_extra = {
            "llm_configs": llm_models,
            "lam": llm_lambda,
            "trainable_lambda": llm_trainable,
            "dataset_description": llm_description,
        }
    else:
        prior = prior_cls
        prior_extra = None

    # --- Setup logging ---
    log_file = setup_logging(cfg)
    log = logging.getLogger("tabularGP")

    log.info("=" * 60)
    log.info(f"TabularGP | Problem: {problem} | Kernel: {kernel_name} | Prior: {prior_name}")
    log.info(f"Config: epochs={epochs}, lr={lr}, noise={noise}, nb_training_points={nb_training_points}")
    log.info(f"Dataset ID: {dataset_id} | n_samples: {n_samples}")
    log.info(f"Log file: {log_file}")
    log.info("=" * 60)

    # --- Load data ---
    df, target, dataset_description = load_openml_dataset(dataset_id, target, n_samples)
    cat_names, cont_names = detect_columns(df, target)
    log.info(f"Categorical features ({len(cat_names)}): {cat_names}")
    log.info(f"Continuous features  ({len(cont_names)}): {cont_names}")
    log.info(f"Dataset shape: {df.shape}")

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

    # For LLMPrior, query LLMs once with dataset summary, then pass predictions to prior
    if prior_extra is not None:
        from tabularGP.prior import LLMPrior
        from tabularGP.llm_utils import query_summary_llms

        log.info("Querying LLMs with dataset summary...")
        # Use OpenML description if config description is empty
        llm_description = prior_extra["dataset_description"] or dataset_description
        llm_predictions = query_summary_llms(
            llm_configs=prior_extra["llm_configs"],
            df=df,
            feature_names=cat_names + cont_names,
            target_name=target,
            dataset_description=llm_description,
            problem_type=problem,
        )
        log.info(f"LLM predictions: {llm_predictions}")

        llm_lam = prior_extra["lam"]
        llm_trainable = prior_extra["trainable_lambda"]
        # Create a factory that passes pre-queried predictions
        class _LLMPriorFactory:
            def __call__(self, train_cat, train_cont, train_out, emb_szs):
                return LLMPrior(train_cat, train_cont, train_out, emb_szs,
                                llm_predictions=llm_predictions, lam=llm_lam,
                                trainable_lambda=llm_trainable)
        prior = _LLMPriorFactory()

    # For LLMKernel, query LLMs for feature weights and create kernel instance
    if kernel_extra is not None:
        from tabularGP.llm_utils import llm_kernel_weights
        from tabularGP.kernel import LLMKernel

        log.info("Querying LLMs for kernel feature weights...")
        llm_k_configs = cfg.get("llm_prior", {}).get("models", [])
        if not llm_k_configs:
            llm_k_configs = [{"model": "nvidia_nim/openai/gpt-oss-120b", "strategy": "statistical"}]
        feature_weights = llm_kernel_weights(
            llm_configs=llm_k_configs,
            df=df,
            feature_names=cat_names + cont_names,
            target_name=target,
            dataset_description=dataset_description,
            problem_type=problem,
        )
        log.info(f"LLM kernel feature weights: {feature_weights}")

        # Pass as a callable factory (TabularGPModel checks isinstance(kernel, type))
        _ke = kernel_extra
        _fw = feature_weights
        # Create a dynamic class so isinstance(kernel, type) is True
        class _LLMKernelConfigured(LLMKernel):
            def __init__(self, train_cat, train_cont, embedding_sizes, **kwargs):
                super().__init__(
                    train_cat, train_cont, embedding_sizes,
                    feature_weights=_fw,
                    base_kernel=_ke["base_kernel"],
                    lam=_ke["lam"],
                    trainable_lambda=_ke["trainable_lambda"],
                )
        kernel = _LLMKernelConfigured

    learn = tabularGP_learner(
        data,
        nb_training_points=nb_training_points,
        kernel=kernel,
        prior=prior,
        noise=noise,
        metrics=metric,
    )


    log.info(f"Training for {epochs} epochs with lr={lr}...")
    learn.fit_one_cycle(epochs, lr_max=lr)

    # --- Log per-epoch metrics from fastai recorder ---
    log.info("")
    log.info("=" * 60)
    log.info("Training Log (per epoch)")
    log.info("=" * 60)
    recorder = learn.recorder
    train_losses = recorder.losses            # all batch losses
    val_losses = recorder.values              # per-epoch: [train_loss, valid_loss, metric, ...]
    metric_name = "accuracy" if problem == "classification" else "rmse"
    log.info(f"{'Epoch':<8} {'Train Loss':<15} {'Valid Loss':<15} {metric_name:<15}")
    log.info("-" * 53)
    for epoch_idx, vals in enumerate(val_losses):
        t_loss = f"{vals[0]:.6f}" if vals[0] is not None else "N/A"
        v_loss = f"{vals[1]:.6f}" if len(vals) > 1 and vals[1] is not None else "N/A"
        m_val  = f"{vals[2]:.6f}" if len(vals) > 2 and vals[2] is not None else "N/A"
        log.info(f"{epoch_idx:<8} {t_loss:<15} {v_loss:<15} {m_val:<15}")

    # --- Log LLMPrior lambda if applicable ---
    if prior_name == "LLMPrior" and hasattr(learn.model.prior, 'lam'):
        log.info(f"\nLLMPrior final λ (learned): {learn.model.prior.lam.item():.4f}")

    # --- Sample Prediction ---
    log.info("")
    log.info("-" * 40)
    log.info("Sample predictions (first 5 validation rows):")
    log.info("-" * 40)
    preds, targets = learn.get_preds(dl=data[1])
    val_idx = splits[1][:5]
    for i in range(min(5, len(preds))):
        pred = preds[i]
        actual = df.iloc[val_idx[i]][target] if i < len(val_idx) else "N/A"
        if problem == "classification":
            pred_label = pred.argmax().item()
            log.info(f"  [{i+1}] Predicted: class {pred_label} | Actual: {actual}")
        else:
            log.info(f"  [{i+1}] Predicted: {pred.item():.4f} | Actual: {actual}")

    # --- Feature Importance ---
    log.info("")
    log.info("-" * 40)
    log.info("Feature Importance:")
    log.info("-" * 40)
    importance = learn.feature_importance.sort_values('Importance', ascending=False)
    log.info("\n" + importance.to_string(index=False))

    if plot_fi:
        learn.plot_feature_importance()
        import matplotlib.pyplot as plt
        plt.savefig("feature_importance.png", bbox_inches="tight", dpi=150)
        log.info("Plot saved to feature_importance.png")

    log.info("")
    log.info("=" * 60)
    log.info(f"Done! Log saved to: {log_file}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
