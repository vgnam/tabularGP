"""
Random Forest Baseline - Compare with TabularGP on same dataset/split
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    problem = cfg["problem"]
    dataset_id = cfg["dataset_id"]
    target = cfg.get("target")
    n_samples = cfg.get("n_samples")

    print(f"Fetching OpenML dataset id={dataset_id}...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    df = data.frame

    if target is None:
        target = data.target_names[0] if isinstance(data.target_names, list) else data.target_names

    df = df.dropna(subset=[target])
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    print(f"Dataset: {data.details.get('name', 'unknown')} | Shape: {df.shape}")
    print(f"Problem: {problem} | Target: {target}")

    # Same split as TabularGP (80/20, seed=42)
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split = int(0.8 * len(df))
    train_idx, val_idx = indices[:split], indices[split:]

    # Prepare features
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Fill NaN
    X = X.fillna(0)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train RF
    if problem == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"\n{'='*50}")
        print(f"Random Forest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"{'='*50}")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val.astype(float), preds))
        print(f"\n{'='*50}")
        print(f"Random Forest RMSE: {rmse_val:.4f}")
        print(f"{'='*50}")

    # Feature importance
    importances = pd.DataFrame({
        "Variable": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    print(f"\nTop 10 Feature Importance:")
    print(importances.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
