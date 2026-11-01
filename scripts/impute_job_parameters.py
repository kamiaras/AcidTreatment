"""Impute missing and clearly invalid job_parameters cells via LR.

Starts from the pre-imputation backup (true source), marks invalid values as
NaN, fits well→job OLS on remaining observed rows, fills gaps, then
recomputes Dosage = 42 * Volume / StimPay when both are valid and > 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
WC_PATH = ROOT / "data" / "processed" / "well_characteristics.xlsx"
JP_PATH = ROOT / "data" / "processed" / "job_parameters.xlsx"
OUT_DIR = ROOT / "outputs" / "well_to_job_lr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer original pre-imputation table so we do not compound earlier fills
SOURCE_CANDIDATES = [
    OUT_DIR / "job_parameters_before_imputation.xlsx",
    JP_PATH,
]

FEATURE_COLS = [
    "Crude",
    "Field",
    "Reservoir",
    "Payzone",
    "WellType",
    "Artificial Lift",
]
MODEL_TARGETS = [
    "Vendor",
    "Type of Intervention",
    "Acid Concentration %",
    "Total Acid Volume",
    "Total Stim Pay ft",
    "Dosage (Gal/ft)",
]
INTEGERISH = {"Vendor", "Type of Intervention"}


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURE_COLS,
            )
        ]
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_miss: pd.DataFrame) -> np.ndarray:
    pipe = make_pipeline()
    pipe.fit(X_train.astype(str), y_train.astype(float))
    return pipe.predict(X_miss.astype(str))


def invalidate_wrong_values(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark clearly invalid cells as NaN. Returns (cleaned, log of invalidated cells)."""
    out = df.copy()
    rows: list[dict] = []

    def nuke(idx: int, col: str, reason: str, old: float) -> None:
        rows.append(
            {
                "Index": int(out.loc[idx, "Index"]),
                "Well": out.loc[idx, "Well"],
                "column": col,
                "old_value": float(old),
                "reason": reason,
            }
        )
        out.loc[idx, col] = np.nan

    # Non-positive continuous design knobs are not physical here
    for col, reason in [
        ("Total Stim Pay ft", "stim_pay_non_positive"),
        ("Total Acid Volume", "acid_volume_non_positive"),
        ("Dosage (Gal/ft)", "dosage_non_positive"),
        ("Acid Concentration %", "concentration_non_positive"),
    ]:
        bad = out[col].notna() & (out[col] <= 0)
        for idx in out.index[bad]:
            nuke(idx, col, reason, float(out.loc[idx, col]))

    # Dosage undefined / meaningless without positive stim pay
    pay_invalid = out["Total Stim Pay ft"].isna() | (out["Total Stim Pay ft"] <= 0)
    dosage_with_bad_pay = out["Dosage (Gal/ft)"].notna() & pay_invalid
    for idx in out.index[dosage_with_bad_pay]:
        nuke(
            idx,
            "Dosage (Gal/ft)",
            "dosage_without_positive_stim_pay",
            float(out.loc[idx, "Dosage (Gal/ft)"]),
        )

    # Dosage must match 42*vol/stim when both inputs are valid and positive
    can_check = (
        out["Total Acid Volume"].notna()
        & (out["Total Acid Volume"] > 0)
        & out["Total Stim Pay ft"].notna()
        & (out["Total Stim Pay ft"] > 0)
        & out["Dosage (Gal/ft)"].notna()
    )
    formula = 42.0 * out["Total Acid Volume"] / out["Total Stim Pay ft"]
    mismatch = can_check & ((out["Dosage (Gal/ft)"] - formula).abs() > 1e-4)
    for idx in out.index[mismatch]:
        nuke(
            idx,
            "Dosage (Gal/ft)",
            "dosage_formula_mismatch",
            float(out.loc[idx, "Dosage (Gal/ft)"]),
        )

    return out, pd.DataFrame(rows)


def load_source() -> pd.DataFrame:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            print(f"Source: {path}")
            return pd.read_excel(path)
    raise FileNotFoundError("No job_parameters source found")


def main() -> None:
    wc = pd.read_excel(WC_PATH)
    jp = load_source()
    if not wc["Index"].equals(jp["Index"]):
        raise ValueError("Index columns do not align between the two tables")

    cleaned, invalid_log = invalidate_wrong_values(jp)
    print(f"Invalidated {len(invalid_log)} clearly-wrong cells")
    if len(invalid_log):
        print(invalid_log.to_string(index=False))

    filled = cleaned.copy()
    X_all = wc[FEATURE_COLS]
    impute_log: list[dict] = []

    for target in MODEL_TARGETS:
        miss = filled[target].isna()
        if not miss.any():
            continue
        observed = filled[target].notna()
        if not observed.any():
            raise ValueError(f"No observed values to train on for {target}")

        y_hat = fit_predict(X_all.loc[observed], filled.loc[observed, target], X_all.loc[miss])
        y_obs = filled.loc[observed, target].astype(float)
        # Stim pay / volume / dosage must stay positive
        if target in {"Total Stim Pay ft", "Total Acid Volume", "Dosage (Gal/ft)", "Acid Concentration %"}:
            lo = max(float(y_obs.min()), 1e-6)
        else:
            lo = float(y_obs.min())
        y_hat = np.clip(y_hat, lo, float(y_obs.max()))
        if target in INTEGERISH:
            y_hat = np.rint(y_hat)

        for idx, pred in zip(filled.index[miss], y_hat):
            impute_log.append(
                {
                    "Index": int(filled.loc[idx, "Index"]),
                    "Well": filled.loc[idx, "Well"],
                    "column": target,
                    "method": "linear_regression",
                    "imputed_value": float(pred),
                }
            )
            filled.loc[idx, target] = float(pred)
        print(f"{target}: filled {int(miss.sum())} via LR")

    # Enforce Dosage = 42 * Volume / StimPay on every row
    vol = filled["Total Acid Volume"]
    pay = filled["Total Stim Pay ft"]
    if not ((vol > 0).all() and (pay > 0).all()):
        raise ValueError("Volume and StimPay must be positive before dosage formula")
    formula_vals = 42.0 * vol / pay
    changed = (filled["Dosage (Gal/ft)"] - formula_vals).abs() > 1e-9
    for idx in filled.index[changed | filled["Dosage (Gal/ft)"].isna()]:
        val = float(formula_vals.loc[idx])
        prior = [
            r
            for r in impute_log
            if r["Index"] == int(filled.loc[idx, "Index"]) and r["column"] == "Dosage (Gal/ft)"
        ]
        if prior:
            prior[0]["method"] = "formula_42_vol_over_stimpay"
            prior[0]["imputed_value"] = val
        else:
            impute_log.append(
                {
                    "Index": int(filled.loc[idx, "Index"]),
                    "Well": filled.loc[idx, "Well"],
                    "column": "Dosage (Gal/ft)",
                    "method": "formula_42_vol_over_stimpay",
                    "imputed_value": val,
                }
            )
        filled.loc[idx, "Dosage (Gal/ft)"] = val
    # Always write exact formula values for all rows (source of truth)
    filled["Dosage (Gal/ft)"] = formula_vals
    print(f"Dosage (Gal/ft): enforced formula on all {len(filled)} rows "
          f"({int(changed.sum())} previously off)")

    remaining = filled[MODEL_TARGETS].isna().sum()
    assert remaining.sum() == 0, remaining
    assert (filled["Total Stim Pay ft"] > 0).all()
    assert (filled["Dosage (Gal/ft)"] > 0).all()
    assert (
        (filled["Dosage (Gal/ft)"] - 42.0 * filled["Total Acid Volume"] / filled["Total Stim Pay ft"])
        .abs()
        .max()
        < 1e-9
    )

    filled.to_excel(JP_PATH, index=False)
    invalid_path = OUT_DIR / "invalidated_cells.csv"
    log_path = OUT_DIR / "imputation_log.csv"
    invalid_log.to_csv(invalid_path, index=False)
    pd.DataFrame(impute_log).sort_values(["Index", "column"]).to_csv(log_path, index=False)

    print(f"\nUpdated: {JP_PATH}")
    print(f"Invalidated log: {invalid_path}")
    print(f"Imputation log:  {log_path}")
    print(f"\nImputed {len(impute_log)} cells:")
    print(pd.DataFrame(impute_log).sort_values(["Index", "column"]).to_string(index=False))


if __name__ == "__main__":
    main()
