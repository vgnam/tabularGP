"""
Benchmark: Compare all 4 Priors in TabularGP (Classification — Accuracy)
=========================================================================
ZeroPrior vs ConstantPrior vs LinearPrior vs LLMPrior

Larger classification datasets:
  1. Adult (#1590, ~48K rows → 5000 subsample, binary)
  2. Bank Marketing (#1461, ~45K rows → 5000 subsample, binary)
  3. Electricity (#44120, ~45K rows → 5000 subsample, binary)
"""

import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_NIM_API_KEY"] = os.environ.get(
    "NVIDIA_NIM_API_KEY",
    "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",
)

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prior_benchmark")

# Tee stdout to file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

_log_file = open("prior_benchmark_results.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from fastai.tabular.all import *
from tabularGP import tabularGP_learner
from tabularGP.kernel import WeightedSumKernel
from tabularGP.prior import ZeroPrior, ConstantPrior, LinearPrior, LLMPrior
from tabularGP.llm_utils import query_summary_llms


# =============================================================================
# LLM config
# =============================================================================
LLM_CONFIGS = [
    {"model": "nvidia_nim/openai/gpt-oss-120b", "strategy": "statistical"},
    {"model": "nvidia_nim/openai/gpt-oss-120b", "strategy": "domain_expert"},
    {"model": "nvidia_nim/openai/gpt-oss-120b", "strategy": "pattern_matching"},
]


# =============================================================================
# Helpers
# =============================================================================
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


def get_llm_predictions(df, cat_names, cont_names, target,
                        dataset_description="", problem_type="classification"):
    feature_names = cat_names + cont_names
    log.info(f"  Querying {len(LLM_CONFIGS)} LLMs...")
    try:
        predictions = query_summary_llms(
            llm_configs=LLM_CONFIGS,
            df=df, feature_names=feature_names,
            target_name=target,
            dataset_description=dataset_description,
            problem_type=problem_type,
        )
        log.info(f"  LLM predictions: {predictions}")
        return predictions
    except Exception as e:
        log.warning(f"  LLM query failed: {e}. Fallback to empty.")
        return []


# =============================================================================
# Train GP classification with a specific prior
# =============================================================================
def run_gp_classification(df, target, cat_names, cont_names,
                          train_idx, val_idx, prior, prior_name,
                          epochs=15, lr=1e-3, noise=0.1):
    df_copy = df.copy()
    for col in cat_names:
        df_copy[col] = df_copy[col].astype(str)
    df_copy[target] = df_copy[target].astype(str)

    procs = [Categorify, FillMissing, Normalize]
    splits = (list(train_idx), list(val_idx))

    to = TabularPandas(
        df_copy, procs=procs,
        cat_names=cat_names, cont_names=cont_names,
        y_names=target, y_block=CategoryBlock(),
        splits=splits,
    )
    data = to.dataloaders(bs=128)
    nb_pts = min(len(train_idx), 500)

    t0 = time.time()
    learn = tabularGP_learner(
        data,
        nb_training_points=nb_pts,
        kernel=WeightedSumKernel,
        prior=prior,
        noise=noise,
        metrics=accuracy,
    )

    with learn.no_bar(), learn.no_logging():
        learn.fit_one_cycle(epochs, lr_max=lr)

    elapsed = time.time() - t0

    preds, targets = learn.get_preds(dl=data[1])
    pred_labels = preds.argmax(dim=-1).numpy()
    true_labels = targets.numpy().astype(int)
    acc = accuracy_score(true_labels, pred_labels)

    if prior_name == "LLMPrior" and hasattr(learn.model.prior, 'lam'):
        log.info(f"    LLMPrior final λ = {learn.model.prior.lam.item():.4f}")

    return acc, elapsed


# =============================================================================
# Build prior configs
# =============================================================================
def build_prior_configs(llm_predictions, lam=0.5):
    priors = {
        "ZeroPrior": ZeroPrior,
        "ConstantPrior": ConstantPrior,
        "LinearPrior": LinearPrior,
    }

    class _LLMPriorFactory:
        def __call__(self, train_cat, train_cont, train_out, emb_szs):
            return LLMPrior(train_cat, train_cont, train_out, emb_szs,
                            llm_predictions=llm_predictions, lam=lam)
    priors["LLMPrior"] = _LLMPriorFactory()
    return priors


# =============================================================================
# Benchmark one dataset
# =============================================================================
def benchmark_dataset(name, df, target, dataset_desc="",
                      epochs=15, lr=1e-3, noise=0.1, n_runs=3):
    df = df.dropna(subset=[target]).reset_index(drop=True)

    if len(df) < 10:
        print(f"\n  ⚠ Skipping '{name}' — only {len(df)} samples.\n")
        return None

    cat_names, cont_names = detect_columns(df, target)
    n_classes = df[target].nunique()

    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"  Samples: {len(df)} | Classes: {n_classes} | Cat: {len(cat_names)}, Cont: {len(cont_names)}")
    print(f"  {n_runs} random splits × 4 priors × {epochs} epochs")
    print(f"{'='*70}")

    llm_preds = get_llm_predictions(df, cat_names, cont_names, target, dataset_desc)
    priors = build_prior_configs(llm_preds)

    results = {pname: {"acc": [], "time": []} for pname in priors}

    for run in range(n_runs):
        seed = 42 + run
        np.random.seed(seed)
        indices = np.random.permutation(len(df))
        split = int(0.8 * len(df))
        train_idx, val_idx = indices[:split], indices[split:]

        print(f"\n  --- Split {run+1}/{n_runs} (seed={seed}) ---")

        for prior_name, prior_obj in priors.items():
            try:
                acc_val, elapsed = run_gp_classification(
                    df, target, cat_names, cont_names,
                    train_idx, val_idx,
                    prior=prior_obj, prior_name=prior_name,
                    epochs=epochs, lr=lr, noise=noise,
                )
                results[prior_name]["acc"].append(acc_val)
                results[prior_name]["time"].append(elapsed)
                print(f"    {prior_name:15s}  Accuracy={acc_val:.4f}  Time={elapsed:.1f}s")
            except Exception as e:
                log.error(f"    {prior_name} FAILED: {e}")
                results[prior_name]["acc"].append(0.0)
                results[prior_name]["time"].append(0)

    # Summary
    print(f"\n  {'─'*60}")
    print(f"  Results for: {name}")
    print(f"  {'─'*60}")
    print(f"  {'Prior':15s}  {'Accuracy (mean±std)':22s}  {'Time (mean)':12s}  {'vs Best':10s}")
    print(f"  {'─'*15}  {'─'*22}  {'─'*12}  {'─'*10}")

    summary_rows = []
    for pname in priors:
        accs = results[pname]["acc"]
        times = results[pname]["time"]
        summary_rows.append({
            "dataset": name, "prior": pname,
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "time_mean": np.mean(times),
        })

    best_acc = max(r["acc_mean"] for r in summary_rows)
    for r in summary_rows:
        diff = ((best_acc - r["acc_mean"]) / best_acc * 100) if best_acc > 0 else 0
        marker = " ★ BEST" if r["acc_mean"] == best_acc else f"-{diff:.1f}%"
        print(f"  {r['prior']:15s}  {r['acc_mean']:.4f} ± {r['acc_std']:.4f}       "
              f"{r['time_mean']:7.1f}s     {marker}")

    return summary_rows


# =============================================================================
# Dataset loaders
# =============================================================================
def load_openml_classification(data_id, name, desc, n_samples=None):
    data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = data.frame
    target = data.target_names[0] if isinstance(data.target_names, list) else data.target_names
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    print(f"  Loaded {name}: {df.shape}, target='{target}', classes={df[target].nunique()}")
    return df, target, desc


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("  PRIOR BENCHMARK (Classification — Accuracy) — Large Datasets")
    print("  ZeroPrior vs ConstantPrior vs LinearPrior vs LLMPrior")
    print("=" * 70)

    all_results = []

    # --- Dataset 1: Adult (#1590, ~48K → 5000, binary) ---
    try:
        df1, t1, d1 = load_openml_classification(
            1590, "Adult (income)",
            "US Census: predict whether income >50K based on demographic features.",
            n_samples=5000)
        res = benchmark_dataset("Adult (#1590, n=5000)", df1, t1, d1,
                                epochs=15, lr=1e-3, noise=0.1, n_runs=3)
        if res:
            all_results.extend(res)
    except Exception as e:
        print(f"\n  ⚠ Adult failed: {e}")

    # --- Dataset 2: Bank Marketing (#1461, ~45K → 5000, binary) ---
    try:
        df2, t2, d2 = load_openml_classification(
            1461, "Bank Marketing",
            "Portuguese bank: predict whether client subscribes to term deposit.",
            n_samples=5000)
        res = benchmark_dataset("Bank Marketing (#1461, n=5000)", df2, t2, d2,
                                epochs=15, lr=1e-3, noise=0.1, n_runs=3)
        if res:
            all_results.extend(res)
    except Exception as e:
        print(f"\n  ⚠ Bank Marketing failed: {e}")

    # --- Dataset 3: Electricity (#44120, ~45K → 5000, binary) ---
    try:
        df3, t3, d3 = load_openml_classification(
            44120, "Electricity",
            "Australian electricity market: predict price movement direction.",
            n_samples=5000)
        res = benchmark_dataset("Electricity (#44120, n=5000)", df3, t3, d3,
                                epochs=15, lr=1e-3, noise=0.1, n_runs=3)
        if res:
            all_results.extend(res)
    except Exception as e:
        print(f"\n  ⚠ Electricity failed: {e}")

    # =====================================================================
    # GRAND SUMMARY
    # =====================================================================
    print(f"\n\n{'='*70}")
    print("  GRAND SUMMARY — All Datasets × All Priors (Accuracy)")
    print(f"{'='*70}")

    if not all_results:
        print("  No results collected.")
        return

    summary_df = pd.DataFrame(all_results)
    datasets = summary_df["dataset"].unique()
    wins = {p: 0 for p in summary_df["prior"].unique()}

    for ds in datasets:
        sub = summary_df[summary_df["dataset"] == ds]
        best_idx = sub["acc_mean"].idxmax()
        winner = sub.loc[best_idx, "prior"]
        wins[winner] += 1
        print(f"\n  [{ds}]")
        for _, row in sub.iterrows():
            marker = " ★" if row["prior"] == winner else ""
            print(f"    {row['prior']:15s}  Acc={row['acc_mean']:.4f} ± {row['acc_std']:.4f}  "
                  f"Time={row['time_mean']:.1f}s{marker}")

    print(f"\n  {'─'*50}")
    print(f"  Overall wins across {len(datasets)} datasets:")
    for prior_name, count in sorted(wins.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"    {prior_name:15s}  {count} win(s)  {bar}")

    print(f"\n  Average Accuracy across all datasets:")
    avg_acc = summary_df.groupby("prior")["acc_mean"].mean().sort_values(ascending=False)
    for prior_name, avg in avg_acc.items():
        print(f"    {prior_name:15s}  avg Accuracy = {avg:.4f}")

    print(f"\n{'='*70}")
    print("  Benchmark complete. Results saved to prior_benchmark_results.txt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
