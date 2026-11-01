"""Confidence-weighted regression: well characteristics -> decline rate b.

Re-fits anchored pre-treatment exponentials

    q(t) = q_{-180} * exp(b * (t + 180))

joins them to well_characteristics.xlsx on Well (ordered for re-treatments),
engineers categorical interaction features, then trains OLS / Ridge / Lasso
with OneHotEncoder, sample_weight=confidence (OLS/Ridge/Lasso), and
GroupKFold by well. Also fits distance-weighted KNN on confidence>0 rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TS_PATH = (
    ROOT / "data" / "raw" / "production_timeseries.csv"
)
WC_PATH = ROOT / "data" / "processed" / "well_characteristics.xlsx"
OUT_DIR = ROOT / "outputs" / "well_to_b_lr"
EXP_FITS_PATH = ROOT / "outputs" / "well_timeseries" / "exp_decline_fits.csv"

CONFIDENCE_K = 3.0
N_SPLITS = 5
BASE_CAT_COLS = [
    "Crude",
    "Field",
    "Reservoir",
    "Payzone",
    "WellType",
    "Artificial Lift",
]
INTERACTION_COLS = [
    "Field_x_Reservoir",
    "Reservoir_x_Payzone",
    "Crude_x_Field",
    "WellType_x_Lift",
]
# treatment_seq is categorical; n_jobs_on_well is numeric
CAT_COLS = BASE_CAT_COLS + INTERACTION_COLS + ["treatment_seq"]
NUM_COLS = ["n_jobs_on_well"]

RIDGE_ALPHAS = np.logspace(-3, 3, 25)
LASSO_ALPHAS = np.logspace(-5, 1, 40)
KNN_KS = (3, 5, 7, 9, 11, 15)


def fit_confidence(r2: float, n: int, k: float = CONFIDENCE_K) -> float:
    if n is None or n < 2 or not np.isfinite(r2):
        return float("nan")
    r2_c = float(np.clip(r2, 0.0, 1.0))
    if n == 2:
        return 0.0
    return r2_c * ((n - 2) / (n - 2 + k))


def fit_anchored_decline(group: pd.DataFrame) -> dict | None:
    """Fit q(t) = q_180 * exp(b*(t+180)); return params or None."""
    anchor = group.loc[group["Index"] == -180, "observation"]
    if len(anchor) == 0:
        return None
    q180 = float(anchor.iloc[0])
    if not np.isfinite(q180) or q180 <= 0:
        return None

    pre = group.loc[
        (group["Index"] < 0) & (group["observation"] > 0),
        ["Index", "observation"],
    ].dropna()
    if len(pre) < 2 or pre["Index"].nunique() < 2:
        return None

    t = pre["Index"].to_numpy(dtype=float)
    q = pre["observation"].to_numpy(dtype=float)
    x = t + 180.0
    y = np.log(q / q180)
    mask = x != 0.0
    if mask.sum() == 0:
        return None

    # OLS through origin: b = (x'y) / (x'x)
    b = float(np.dot(x[mask], y[mask]) / np.dot(x[mask], x[mask]))
    a = float(q180 * np.exp(180.0 * b))
    pred = q180 * np.exp(b * x)
    ss_res = float(np.sum((q - pred) ** 2))
    ss_tot = float(np.sum((q - q.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = int(len(pre))
    return {
        "ID": int(group["ID"].iloc[0]),
        "Well": str(group["Well"].iloc[0]),
        "q_neg180": q180,
        "a": a,
        "b": b,
        "n": n,
        "r2": r2,
        "confidence": fit_confidence(r2, n),
    }


def compute_anchored_fits(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in ts.groupby("ID", sort=True):
        fit = fit_anchored_decline(group)
        if fit is not None:
            rows.append(fit)
    fits = pd.DataFrame(rows)
    fits["confidence_rank"] = (
        fits["confidence"].rank(ascending=False, method="min").astype("Int64")
    )
    return fits


def join_fits_to_well_chars(fits: pd.DataFrame, wc: pd.DataFrame) -> pd.DataFrame:
    """Align fit rows to WC rows on Well; zip by Index/ID order within well."""
    pairs: list[tuple[int, int]] = []  # (fit_idx, wc_idx)
    for well, fit_g in fits.groupby("Well", sort=False):
        wc_g = wc.loc[wc["Well"] == well]
        if wc_g.empty:
            continue
        fit_sorted = fit_g.sort_values("ID")
        wc_sorted = wc_g.sort_values("Index")
        n = min(len(fit_sorted), len(wc_sorted))
        for fi, wi in zip(fit_sorted.index[:n], wc_sorted.index[:n]):
            pairs.append((fi, wi))

    if not pairs:
        raise ValueError("No Well overlaps between fits and well_characteristics")

    fit_ix, wc_ix = zip(*pairs)
    left = fits.loc[list(fit_ix)].reset_index(drop=True)
    right = wc.loc[list(wc_ix)].reset_index(drop=True)
    # Prefer fit Well/ID; keep WC Index as Index_wc
    right = right.drop(columns=["Well"]).rename(columns={"Index": "Index_wc"})
    merged = pd.concat([left, right], axis=1)
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in BASE_CAT_COLS:
        out[c] = out[c].astype(str)
    out["Field_x_Reservoir"] = out["Field"] + "_x_" + out["Reservoir"]
    out["Reservoir_x_Payzone"] = out["Reservoir"] + "_x_" + out["Payzone"]
    out["Crude_x_Field"] = out["Crude"] + "_x_" + out["Field"]
    out["WellType_x_Lift"] = (
        out["WellType"] + "_x_" + out["Artificial Lift"]
    )

    # Sequence within well using WC Index order already reflected in join order;
    # recompute from Index_wc for clarity.
    out = out.sort_values(["Well", "Index_wc"])
    out["treatment_seq"] = (
        out.groupby("Well").cumcount() + 1
    ).astype(str)
    out["n_jobs_on_well"] = out.groupby("Well")["Well"].transform("size").astype(
        float
    )
    return out.sort_values("ID").reset_index(drop=True)


def make_ols_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    return Pipeline([("pre", pre), ("model", LinearRegression())])


def make_regularized_pipeline(model) -> Pipeline:
    """One-hot categoricals, passthrough numerics, then scale all for penalty."""
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            ),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def make_ridge_pipeline() -> Pipeline:
    return make_regularized_pipeline(RidgeCV(alphas=RIDGE_ALPHAS))


def make_lasso_pipeline() -> Pipeline:
    return make_regularized_pipeline(
        LassoCV(alphas=LASSO_ALPHAS, max_iter=20000, random_state=0)
    )


def make_knn_pipeline(n_neighbors: int = 5) -> Pipeline:
    """One-hot + scale; distance-weighted KNN (fit on confidence>0 only)."""
    return make_regularized_pipeline(
        KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="euclidean",
        )
    )


def _weighted_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    mask = w > 0
    if mask.sum() == 0:
        return float("nan")
    return float(r2_score(y[mask], yhat[mask], sample_weight=w[mask]))


def _weighted_mae(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    mask = w > 0
    if mask.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y[mask], yhat[mask], sample_weight=w[mask]))


def _weighted_rmse(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    mask = w > 0
    if mask.sum() == 0:
        return float("nan")
    return float(
        np.sqrt(
            mean_squared_error(y[mask], yhat[mask], sample_weight=w[mask])
        )
    )


def oof_predict(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    weights: np.ndarray,
    *,
    weighted_fit: bool = True,
    fit_weight_gt0_only: bool = False,
) -> tuple[np.ndarray, int]:
    n_groups = int(groups.nunique())
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError("Not enough well groups for GroupKFold")
    gkf = GroupKFold(n_splits=n_splits)
    yhat = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold_pipe = clone(pipe)
        if fit_weight_gt0_only:
            keep = train_idx[weights[train_idx] > 0]
            if len(keep) == 0:
                continue
            # Shrink k if train set is smaller than n_neighbors
            model = fold_pipe.named_steps["model"]
            if hasattr(model, "n_neighbors"):
                model.set_params(n_neighbors=min(model.n_neighbors, len(keep)))
            fold_pipe.fit(X.iloc[keep], y[keep])
        elif weighted_fit:
            fold_pipe.fit(
                X.iloc[train_idx],
                y[train_idx],
                model__sample_weight=weights[train_idx],
            )
        else:
            fold_pipe.fit(X.iloc[train_idx], y[train_idx])
        yhat[test_idx] = fold_pipe.predict(X.iloc[test_idx])
    return yhat, n_splits


def feature_names_from_pipe(pipe: Pipeline) -> np.ndarray:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    return pre.get_feature_names_out()


def model_coefficients(pipe: Pipeline, name: str) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return None
    names = feature_names_from_pipe(pipe)
    coef = model.coef_
    if "scale" in pipe.named_steps:
        scaler: StandardScaler = pipe.named_steps["scale"]
        coef = coef / scaler.scale_
    return (
        pd.DataFrame({"feature": names, "coefficient": coef})
        .assign(model=name, intercept=float(model.intercept_))
        .sort_values("coefficient", key=np.abs, ascending=False)
    )


def eval_model(
    name: str,
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    weights: np.ndarray,
    *,
    weighted_fit: bool = True,
    fit_weight_gt0_only: bool = False,
) -> tuple[dict, pd.DataFrame | None, np.ndarray, np.ndarray]:
    pipe_in = clone(pipe)
    if fit_weight_gt0_only:
        keep = np.flatnonzero(weights > 0)
        model = pipe_in.named_steps["model"]
        if hasattr(model, "n_neighbors"):
            model.set_params(n_neighbors=min(model.n_neighbors, len(keep)))
        pipe_in.fit(X.iloc[keep], y[keep])
    elif weighted_fit:
        pipe_in.fit(X, y, model__sample_weight=weights)
    else:
        pipe_in.fit(X, y)
    yhat_in = pipe_in.predict(X)
    yhat_oof, n_splits = oof_predict(
        pipe,
        X,
        y,
        groups,
        weights,
        weighted_fit=weighted_fit,
        fit_weight_gt0_only=fit_weight_gt0_only,
    )

    model = pipe_in.named_steps["model"]
    alpha = getattr(model, "alpha_", np.nan)
    n_neighbors = getattr(model, "n_neighbors", np.nan)
    metrics = {
        "model": name,
        "n": int(len(y)),
        "n_weight_gt0": int((weights > 0).sum()),
        "n_wells": int(groups.nunique()),
        "n_splits": int(n_splits),
        "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "n_neighbors": int(n_neighbors) if np.isfinite(n_neighbors) else np.nan,
        "r2_in_sample": float(r2_score(y, yhat_in, sample_weight=weights)),
        "r2_oof_weighted": _weighted_r2(y, yhat_oof, weights),
        "r2_oof_unweighted": float(r2_score(y, yhat_oof)),
        "mae_in_sample": float(
            mean_absolute_error(y, yhat_in, sample_weight=weights)
        ),
        "mae_oof_weighted": _weighted_mae(y, yhat_oof, weights),
        "mae_oof_unweighted": float(mean_absolute_error(y, yhat_oof)),
        "rmse_oof_weighted": _weighted_rmse(y, yhat_oof, weights),
        "rmse_oof_unweighted": float(np.sqrt(mean_squared_error(y, yhat_oof))),
        "y_mean": float(np.average(y, weights=np.maximum(weights, 0))),
        "y_std": float(y.std(ddof=0)),
    }
    coefs = model_coefficients(pipe_in, name)
    return metrics, coefs, yhat_in, yhat_oof


def select_knn_k(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    weights: np.ndarray,
) -> tuple[int, dict[int, float]]:
    """Pick k by best GroupKFold weighted OOF R² (fit on confidence>0 only)."""
    scores: dict[int, float] = {}
    for k in KNN_KS:
        pipe = make_knn_pipeline(k)
        yhat, _ = oof_predict(
            pipe,
            X,
            y,
            groups,
            weights,
            weighted_fit=False,
            fit_weight_gt0_only=True,
        )
        scores[k] = _weighted_r2(y, yhat, weights)
    best_k = max(scores, key=lambda kk: scores[kk] if np.isfinite(scores[kk]) else -np.inf)
    return best_k, scores


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.read_csv(TS_PATH)
    wc = pd.read_excel(WC_PATH)
    fits = compute_anchored_fits(ts)
    fits_path = OUT_DIR / "anchored_fits.csv"
    fits.to_csv(fits_path, index=False)
    print(f"Wrote {len(fits)} anchored fits -> {fits_path}")

    merged = join_fits_to_well_chars(fits, wc)
    merged = engineer_features(merged)
    print(
        f"Joined {len(merged)} jobs; "
        f"weight>0: {(merged['confidence'].fillna(0) > 0).sum()}"
    )

    X = merged[CAT_COLS + NUM_COLS].copy()
    y = merged["b"].to_numpy(dtype=float)
    weights = merged["confidence"].fillna(0.0).to_numpy(dtype=float)
    groups = merged["Well"]

    best_k, knn_scores = select_knn_k(X, y, groups, weights)
    print(
        "KNN k search (weighted OOF R²): "
        + ", ".join(f"k={k}:{s:.3f}" for k, s in knn_scores.items())
        + f" -> best k={best_k}"
    )

    models: dict[str, tuple[Pipeline, dict]] = {
        "ols": (make_ols_pipeline(), {"weighted_fit": True}),
        "ridge": (make_ridge_pipeline(), {"weighted_fit": True}),
        "lasso": (make_lasso_pipeline(), {"weighted_fit": True}),
        "knn": (
            make_knn_pipeline(best_k),
            {"weighted_fit": False, "fit_weight_gt0_only": True},
        ),
    }

    metrics_rows = []
    pred_frame = merged[
        ["ID", "Well", "Index_wc", "b", "confidence", "n", "r2"]
    ].copy()
    all_coefs = []

    for name, (pipe, fit_kw) in models.items():
        metrics, coefs, yhat_in, yhat_oof = eval_model(
            name, pipe, X, y, groups, weights, **fit_kw
        )
        metrics_rows.append(metrics)
        pred_frame[f"{name}_in_sample"] = yhat_in
        pred_frame[f"{name}_oof"] = yhat_oof
        if coefs is not None:
            all_coefs.append(coefs)
            coefs.to_csv(OUT_DIR / f"coefficients_{name}.csv", index=False)
        print(
            f"{name}: OOF R²(w)={metrics['r2_oof_weighted']:.3f} "
            f"R²={metrics['r2_oof_unweighted']:.3f} "
            f"MAE(w)={metrics['mae_oof_weighted']:.4g} "
            f"alpha={metrics['alpha']} k={metrics['n_neighbors']}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    pred_frame.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    if all_coefs:
        pd.concat(all_coefs, ignore_index=True).to_csv(
            OUT_DIR / "coefficients_all.csv", index=False
        )
    pd.DataFrame(
        [{"k": k, "r2_oof_weighted": s} for k, s in knn_scores.items()]
    ).to_csv(OUT_DIR / "knn_k_search.csv", index=False)

    # Attach ridge predictions (+ anchored b) onto the timeseries exp_decline_fits table
    if EXP_FITS_PATH.exists():
        exp_fits = pd.read_csv(EXP_FITS_PATH)
        drop = [
            c
            for c in exp_fits.columns
            if c.startswith("ridge_") or c == "b_anchored"
        ]
        exp_fits = exp_fits.drop(columns=drop, errors="ignore")
        ridge_cols = pred_frame[["ID", "b", "ridge_in_sample", "ridge_oof"]].rename(
            columns={
                "b": "b_anchored",
                "ridge_in_sample": "ridge_b_in_sample",
                "ridge_oof": "ridge_b_oof",
            }
        )
        exp_fits = exp_fits.merge(ridge_cols, on="ID", how="left")
        exp_fits.to_csv(EXP_FITS_PATH, index=False)
        print(f"Updated ridge columns on {EXP_FITS_PATH}")

    print(f"Wrote metrics / OOF / coefficients under {OUT_DIR}")


if __name__ == "__main__":
    main()
