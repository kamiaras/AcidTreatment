"""Linear regression: well characteristics -> job parameters.

Predicts each column in job_parameters.xlsx from well_characteristics.xlsx
using OLS with one-hot-encoded categoricals and GroupKFold CV by well.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
WC_PATH = ROOT / "data" / "processed" / "well_characteristics.xlsx"
JP_PATH = ROOT / "data" / "processed" / "job_parameters.xlsx"
OUT_DIR = ROOT / "outputs" / "well_to_job_lr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Crude",
    "Field",
    "Reservoir",
    "Payzone",
    "WellType",
    "Artificial Lift",
]
TARGET_COLS = [
    "Vendor",
    "Type of Intervention",
    "Acid Concentration %",
    "Total Acid Volume",
    "Total Stim Pay ft",
    "Dosage (Gal/ft)",
]
N_SPLITS = 5


def load_merged() -> pd.DataFrame:
    wc = pd.read_excel(WC_PATH)
    jp = pd.read_excel(JP_PATH)
    if not wc["Index"].equals(jp["Index"]):
        raise ValueError("Index columns do not align between the two tables")
    merged = wc.merge(jp.drop(columns=["Well"]), on="Index", how="inner")
    return merged


def make_pipeline() -> Pipeline:
    # All well features are integer-coded categoricals
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURE_COLS,
            )
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("model", LinearRegression()),
        ]
    )


def fit_report_target(
    df: pd.DataFrame, target: str
) -> tuple[dict, pd.DataFrame, pd.Series]:
    mask = df[target].notna()
    # StimPay 0 makes dosage undefined / degenerate; keep row if target itself is present
    sub = df.loc[mask].copy()
    X = sub[FEATURE_COLS].astype(str)  # force categorical treatment
    y = sub[target].astype(float)
    groups = sub["Well"]

    pipe = make_pipeline()
    pipe.fit(X, y)
    y_hat_in = pipe.predict(X)
    n_groups = groups.nunique()
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError(f"Not enough well groups for CV on {target}")

    gkf = GroupKFold(n_splits=n_splits)
    y_hat_oof = cross_val_predict(pipe, X, y, cv=gkf, groups=groups)

    metrics = {
        "target": target,
        "n": int(len(y)),
        "n_wells": int(n_groups),
        "n_splits": n_splits,
        "r2_in_sample": float(r2_score(y, y_hat_in)),
        "r2_oof": float(r2_score(y, y_hat_oof)),
        "mae_in_sample": float(mean_absolute_error(y, y_hat_in)),
        "mae_oof": float(mean_absolute_error(y, y_hat_oof)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std(ddof=0)),
    }

    # Coefficient table (one-hot feature names)
    ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]
    feat_names = ohe.get_feature_names_out(FEATURE_COLS)
    coefs = pd.DataFrame(
        {
            "feature": feat_names,
            "coefficient": pipe.named_steps["model"].coef_,
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefs.insert(0, "target", target)
    coefs["intercept"] = pipe.named_steps["model"].intercept_

    preds = pd.Series(y_hat_oof, index=sub.index, name=f"{target}_oof_pred")
    return metrics, coefs, preds


def main() -> None:
    df = load_merged()
    print(f"Merged rows: {len(df)}  wells: {df['Well'].nunique()}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Targets:  {TARGET_COLS}\n")

    all_metrics: list[dict] = []
    all_coefs: list[pd.DataFrame] = []
    pred_frame = df[["Index", "Well"]].copy()

    for target in TARGET_COLS:
        metrics, coefs, preds = fit_report_target(df, target)
        all_metrics.append(metrics)
        all_coefs.append(coefs)
        pred_frame.loc[preds.index, preds.name] = preds.values
        pred_frame.loc[preds.index, target] = df.loc[preds.index, target].values

        print(
            f"{target:25s}  n={metrics['n']:3d}  "
            f"R2_in={metrics['r2_in_sample']:+.3f}  "
            f"R2_oof={metrics['r2_oof']:+.3f}  "
            f"MAE_oof={metrics['mae_oof']:.3f}"
        )

    metrics_df = pd.DataFrame(all_metrics)
    coefs_df = pd.concat(all_coefs, ignore_index=True)

    metrics_path = OUT_DIR / "metrics.csv"
    coefs_path = OUT_DIR / "coefficients.csv"
    preds_path = OUT_DIR / "oof_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    coefs_df.to_csv(coefs_path, index=False)
    pred_frame.to_csv(preds_path, index=False)

    print(f"\nWrote:\n  {metrics_path}\n  {coefs_path}\n  {preds_path}")
    print("\n=== Summary ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
