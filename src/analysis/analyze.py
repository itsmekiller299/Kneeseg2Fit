#!/usr/bin/env python
"""
Stage 4 — Analysis.

Merge measurements with patient metadata into a pandas dataframe.
Run basic stats: t-test/ANOVA comparing meniscus thickness across OA vs non-OA and male vs female.
Train a simple classifier (logistic regression and/or random forest) predicting OA presence
from thickness + age + sex; report accuracy/AUC on a held-out split.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(measurements_path, metadata_path):
    """Load measurements and metadata, merge into a single DataFrame."""
    measurements = pd.read_csv(measurements_path)
    metadata = pd.read_csv(metadata_path)
    df = pd.merge(measurements, metadata, on="subject_id", how="inner")
    return df


def basic_stats(df):
    """Run t-tests/ANOVA comparing measurements across OA status and sex."""
    results = {}

    # Meniscus thickness across KL grade (ordinal: 0,1,2,3,4)
    if "kl_grade" in df.columns and "meniscus_thickness_mean_px" in df.columns:
        groups = [
            df[df["kl_grade"] == k]["meniscus_thickness_mean_px"]
            for k in sorted(df["kl_grade"].unique())
        ]
        if len(groups) >= 2 and all(g.size > 0 for g in groups):
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                results["meniscus_thickness_vs_kl_fanova"] = {
                    "f_stat": float(f_stat),
                    "p_val": float(p_val),
                }
            except Exception:
                pass

        # t-test: OA (KL >= 2) vs non-OA (KL < 2)
        df["oa_status"] = df["kl_grade"].apply(lambda g: 1 if g >= 2 else 0)
        oa_thick = df[df["oa_status"] == 1]["meniscus_thickness_mean_px"]
        non_oa_thick = df[df["oa_status"] == 0]["meniscus_thickness_mean_px"]
        if len(oa_thick) > 0 and len(non_oa_thick) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(oa_thick, non_oa_thick)
                results["meniscus_thickness_vs_oa_ttest"] = {
                    "t_stat": float(t_stat),
                    "p_val": float(p_val),
                }
            except Exception:
                pass

    # Meniscus thickness across sex
    if "sex" in df.columns and "meniscus_thickness_mean_px" in df.columns:
        # sex: 0=Female, 1=Male (from our mock metadata)
        male_thick = df[df["sex"] == 1]["meniscus_thickness_mean_px"]
        female_thick = df[df["sex"] == 0]["meniscus_thickness_mean_px"]
        if len(male_thick) > 0 and len(female_thick) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(male_thick, female_thick)
                results["meniscus_thickness_vs_sex_ttest"] = {
                    "t_stat": float(t_stat),
                    "p_val": float(p_val),
                }
            except Exception:
                pass

    return results


def train_classifier(df, feature_cols=None, test_size=0.3, random_state=42):
    """Train logistic regression and random forest to predict OA from measurements."""
    if feature_cols is None:
        feature_cols = ["meniscus_thickness_mean_px", "age", "sex"]

    # Prepare data
    X = df[feature_cols].copy()
    y = df["oa_status"]  # must have oa_status column

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
    logreg.fit(X_train, y_train)
    y_logreg_pred = logreg.predict(X_test)
    try:
        logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
    except Exception:
        logreg_auc = None
    logreg_acc = accuracy_score(y_test, y_logreg_pred)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_rf_pred = rf.predict(X_test)
    try:
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    except Exception:
        rf_auc = None
    rf_acc = accuracy_score(y_test, y_rf_pred)

    results = {
        "logreg_accuracy": float(logreg_acc),
        "logreg_auc": float(logreg_auc) if logreg_auc is not None else None,
        "rf_accuracy": float(rf_acc),
        "rf_auc": float(rf_auc) if rf_auc is not None else None,
        "feature_cols": feature_cols,
        "test_size": test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return results, X_test, y_test, logreg, rf


def plot_volcano(df, x_col="meniscus_thickness_mean_px", group_col="oa_status"):
    """Simple plot of measurement distribution by group."""
    # Boxplot-style summary
    groups = df.groupby(group_col)[x_col]
    return groups


def main(
    measurements_path="data/measurements.csv",
    metadata_path="data/mock_oai/metadata.csv",
):
    """Run full Stage 4 analysis."""
    # First, load and merge data
    # If files don't exist, create synthetic example for demonstration
    if not os.path.exists(measurements_path) or not os.path.exists(metadata_path):
        print("Generating synthetic measurements for demonstration...")
        # Load mock metadata
        meta = pd.read_csv(metadata_path)

        # Create synthetic measurements per subject
        np.random.seed(42)
        records = []
        for _, row in meta.iterrows():
            subj_id = row["subject_id"]
            # Synthetic meniscus thickness: baseline + OA effect + noise
            base_thickness = np.random.uniform(8, 15)
            oa_effect = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% have OA
            kl = row["kl_grade"]
            thickness = (
                base_thickness + (5 if oa_effect else 0) + np.random.normal(0, 1.5)
            )
            age = row["age"]
            sex = row["sex"]

            records.append(
                {
                    "subject_id": subj_id,
                    "meniscus_thickness_mean_px": round(float(thickness), 2),
                    "femoral_width_px": round(float(np.random.uniform(80, 110)), 2),
                    "femoral_ap_px": round(float(np.random.uniform(50, 80)), 2),
                    "tibial_width_px": round(float(np.random.uniform(85, 120)), 2),
                    "tibial_ap_px": round(float(np.random.uniform(45, 75)), 2),
                    "tibial_area_px2": round(float(np.random.uniform(3000, 5000)), 2),
                    "kl_grade": kl,
                    "age": age,
                    "sex": sex,
                    "oa_status": 1 if kl >= 2 else 0,
                }
            )

        df = pd.DataFrame(records)
        measurements_out = "data/measurements_demo.csv"
        df.to_csv(measurements_out, index=False)
        print(f"Wrote {measurements_out}")
    else:
        df = load_data(measurements_path, metadata_path)

    print(f"\nData loaded: {len(df)} patients")
    print(df.head())

    # Basic stats
    stats_results = basic_stats(df)
    print("\n--- Basic Statistics ---")
    for name, res in stats_results.items():
        print(f"{name}: {res}")

    # Train classifier (requires oa_status column)
    if "oa_status" in df.columns:
        print("\n--- Classifier Training ---")
        results, X_test, y_test, logreg, rf = train_classifier(df)
        print(f"Logistic Regression: accuracy={results['logreg_accuracy']:.3f}", end="")
        if results["logreg_auc"] is not None:
            print(f", AUC={results['logreg_auc']:.3f}", end="")
        print()
        print(f"Random Forest: accuracy={results['rf_accuracy']:.3f}", end="")
        if results["rf_auc"] is not None:
            print(f", AUC={results['rf_auc']:.3f}", end="")
        print()

        # Feature importances from RF
        if "rf_accuracy" in dir():
            importances = pd.DataFrame(
                {
                    "feature": results["feature_cols"],
                    "importance": rf.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            print("\nRandom Forest feature importances:")
            print(importances)
    else:
        print("\nCannot train classifier: 'oa_status' column missing.")
        print("Add oa_status to your data or let the demo generate it.")


if __name__ == "__main__":
    import os

    main()
