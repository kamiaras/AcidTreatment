"""Regression pipeline: job / well / decline features -> oil targets.

Joins (inner only; rows missing from any source are dropped):

  - data/processed/job_parameters.xlsx
  - data/processed/well_characteristics.xlsx
  - outputs/well_timeseries/exp_decline_fits.csv
  - outputs/well_timeseries/observed_oil_0_180.csv

Join rule (same as well→b): never equate timeseries ID with job-table Index.
Align on Well; for re-treatments zip fit rows (ID order) with job/well rows
(Index order).

Targets (--target):
  absolute     oil_0_180_observed
               -> outputs/observed_oil_regression/
  relative     (oil_0_180_observed - oil_0_180_no_shutdown) / oil_0_180_no_shutdown
               -> outputs/relative_uplift_regression/
               Keeps oil_0_180_no_shutdown as a feature (level / interaction base).
  volume_norm  (oil_0_180_observed - oil_0_180_no_shutdown) / Total Acid Volume
               -> outputs/uplift_per_acid_volume_regression/
               Keeps both oil_0_180_no_shutdown and Total Acid Volume as features
               in oil/job-inclusive sets (level / scale controls, not stripped).

Also engineers categorical interactions, numeric products, ratios, and
group-mean deviations, and sweeps feature sets that isolate those families.

Sweeps many feature sets × models under GroupKFold by well.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
JP_PATH = ROOT / "data" / "processed" / "job_parameters.xlsx"
WC_PATH = ROOT / "data" / "processed" / "well_characteristics.xlsx"
FITS_PATH = ROOT / "outputs" / "well_timeseries" / "exp_decline_fits.csv"
OBS_PATH = ROOT / "outputs" / "well_timeseries" / "observed_oil_0_180.csv"

N_SPLITS = 5
RANDOM_STATE = 0
OBS_COL = "oil_0_180_observed"
CF_COL = "oil_0_180_no_shutdown"
VOL_COL = "Total Acid Volume"
REL_TARGET = "relative_uplift"
VOL_TARGET = "uplift_per_acid_volume"

WELL_CAT = [
    "Crude",
    "Field",
    "Reservoir",
    "Payzone",
    "WellType",
    "Artificial Lift",
]
WELL_INTERACTION = [
    "Field_x_Reservoir",
    "Reservoir_x_Payzone",
    "Crude_x_Field",
    "WellType_x_Lift",
]
JOB_CAT = ["Vendor", "Type of Intervention"]
JOB_NUM = [
    "Acid Concentration %",
    "Total Acid Volume",
    "Total Stim Pay ft",
    "Dosage (Gal/ft)",
]
DECLINE_NUM = [
    "a",
    "b",
    "n",
    "r2",
    "confidence",
    "b_anchored",
    "ridge_b_oof",
    "a_hat",
]
PRE_OIL_NUM = ["oil_neg180_0_observed"]
CF_RIDGE_NUM = ["oil_0_180_no_shutdown"]
CF_FREE_NUM = ["oil_0_180_no_shutdown_free"]
SEQ_CAT = ["treatment_seq"]
SEQ_NUM = ["n_jobs_on_well"]
LOG_NUM = [
    "log_Total_Acid_Volume",
    "log_Total_Stim_Pay_ft",
    "log_Dosage",
    "log_oil_neg180_0_observed",
    "log_oil_0_180_no_shutdown",
]
# Numeric products / ratios / deviations (built in engineer_features)
PRODUCT_NUM = [
    "conc_x_dosage",
    "conc_x_volume",
    "dosage_x_stimpay",
    "volume_x_stimpay",
    "dosage_x_log_cf",
    "conc_x_log_cf",
    "dosage_x_log_pre",
    "conc_x_log_pre",
    "b_x_log_cf",
    "ridge_b_x_log_cf",
    "n_x_confidence",
    "a_hat_x_exp_neg_b180",  # rough scale proxy related to decline
]
RATIO_NUM = [
    "pre_over_cf",
    "free_cf_over_cf",
    "dosage_over_conc",
    "volume_over_stimpay",  # ~ dosage in gal/ft if units align; keep anyway
    "a_over_a_hat",
]
DEVIATION_NUM = [
    "dosage_dev_field",
    "conc_dev_field",
    "volume_dev_field",
    "dosage_dev_reservoir",
    "log_cf_dev_field",
    "log_pre_dev_field",
    "dosage_z_global",
    "log_cf_z_global",
]
ENGINEERED_CAT = [
    "Vendor_x_Type",
    "Field_x_WellType",
    "Field_x_Vendor",
    "Reservoir_x_Lift",
    "WellType_x_Vendor",
]

# Modest grids — RidgeCV is cheap; avoid heavy ElasticNetCV sweeps.
RIDGE_ALPHAS = np.logspace(-2, 3, 12)
LASSO_ALPHAS = np.logspace(-3, 3, 12)


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Fit an inner regressor on log1p(y); predict in original space."""

    def __init__(self, estimator: BaseEstimator | None = None):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        y = np.asarray(y, dtype=float)
        if (y < 0).any():
            raise ValueError("LogTargetRegressor requires non-negative y")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.log1p(y), **fit_params)
        return self

    def predict(self, X):
        return np.expm1(self.estimator_.predict(X))


class AsinhTargetRegressor(BaseEstimator, RegressorMixin):
    """Fit on asinh(y); works with negative targets (relative uplift)."""

    def __init__(self, estimator: BaseEstimator | None = None):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.arcsinh(np.asarray(y, dtype=float)), **fit_params)
        return self

    def predict(self, X):
        return np.sinh(self.estimator_.predict(X))


def zip_join_on_well(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    well_col: str = "Well",
    left_order: str,
    right_order: str,
) -> pd.DataFrame:
    """Pair rows by Well, zipping sorted left/right for re-treatments."""
    pairs: list[tuple[int, int]] = []
    for well, left_g in left.groupby(well_col, sort=False):
        right_g = right.loc[right[well_col] == well]
        if right_g.empty:
            continue
        left_sorted = left_g.sort_values(left_order)
        right_sorted = right_g.sort_values(right_order)
        n = min(len(left_sorted), len(right_sorted))
        for li, ri in zip(left_sorted.index[:n], right_sorted.index[:n]):
            pairs.append((li, ri))
    if not pairs:
        raise ValueError("No Well overlaps for zip join")
    left_ix, right_ix = zip(*pairs)
    L = left.loc[list(left_ix)].reset_index(drop=True)
    R = right.loc[list(right_ix)].reset_index(drop=True)
    drop_cols = [c for c in R.columns if c in L.columns]
    R = R.drop(columns=drop_cols)
    return pd.concat([L, R], axis=1)


def load_merged() -> pd.DataFrame:
    jp = pd.read_excel(JP_PATH)
    wc = pd.read_excel(WC_PATH)
    fits = pd.read_csv(FITS_PATH)
    obs = pd.read_csv(OBS_PATH)

    # Timeseries tables share ID; job tables share Index. Drop rows not in both.
    fo = fits.merge(obs.drop(columns=["Well"]), on="ID", how="inner")
    jw = wc.merge(jp.drop(columns=["Well"]), on="Index", how="inner")
    merged = zip_join_on_well(
        fo, jw, left_order="ID", right_order="Index"
    )
    return merged.sort_values("ID").reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in WELL_CAT + JOB_CAT:
        out[c] = out[c].astype(str)
    out["Field_x_Reservoir"] = out["Field"] + "_x_" + out["Reservoir"]
    out["Reservoir_x_Payzone"] = out["Reservoir"] + "_x_" + out["Payzone"]
    out["Crude_x_Field"] = out["Crude"] + "_x_" + out["Field"]
    out["WellType_x_Lift"] = out["WellType"] + "_x_" + out["Artificial Lift"]
    out["Vendor_x_Type"] = (
        out["Vendor"] + "_x_" + out["Type of Intervention"]
    )
    out["Field_x_WellType"] = out["Field"] + "_x_" + out["WellType"]
    out["Field_x_Vendor"] = out["Field"] + "_x_" + out["Vendor"]
    out["Reservoir_x_Lift"] = out["Reservoir"] + "_x_" + out["Artificial Lift"]
    out["WellType_x_Vendor"] = out["WellType"] + "_x_" + out["Vendor"]

    out = out.sort_values(["Well", "Index"])
    out["treatment_seq"] = (out.groupby("Well").cumcount() + 1).astype(str)
    out["n_jobs_on_well"] = (
        out.groupby("Well")["Well"].transform("size").astype(float)
    )

    conc = out["Acid Concentration %"].astype(float)
    vol = out["Total Acid Volume"].astype(float)
    stim = out["Total Stim Pay ft"].astype(float)
    dosage = out["Dosage (Gal/ft)"].astype(float)
    pre = out["oil_neg180_0_observed"].astype(float)
    cf = out["oil_0_180_no_shutdown"].astype(float)
    cf_free = out["oil_0_180_no_shutdown_free"].astype(float)
    b = out["b"].astype(float)
    ridge_b = out["ridge_b_oof"].astype(float)
    a = out["a"].astype(float)
    a_hat = out["a_hat"].astype(float)
    n_pts = out["n"].astype(float)
    conf = out["confidence"].astype(float)

    out["log_Total_Acid_Volume"] = np.log1p(vol.clip(lower=0))
    out["log_Total_Stim_Pay_ft"] = np.log1p(stim.clip(lower=0))
    out["log_Dosage"] = np.log1p(dosage.clip(lower=0))
    out["log_oil_neg180_0_observed"] = np.log1p(pre.clip(lower=0))
    out["log_oil_0_180_no_shutdown"] = np.log1p(cf.clip(lower=0))
    log_cf = out["log_oil_0_180_no_shutdown"]
    log_pre = out["log_oil_neg180_0_observed"]

    # Products / multiplications
    out["conc_x_dosage"] = conc * dosage
    out["conc_x_volume"] = conc * vol
    out["dosage_x_stimpay"] = dosage * stim
    out["volume_x_stimpay"] = vol * stim
    out["dosage_x_log_cf"] = dosage * log_cf
    out["conc_x_log_cf"] = conc * log_cf
    out["dosage_x_log_pre"] = dosage * log_pre
    out["conc_x_log_pre"] = conc * log_pre
    out["b_x_log_cf"] = b * log_cf
    out["ridge_b_x_log_cf"] = ridge_b * log_cf
    out["n_x_confidence"] = n_pts * conf.fillna(0.0)
    out["a_hat_x_exp_neg_b180"] = a_hat * np.exp(-ridge_b * 180.0)

    # Ratios
    out["pre_over_cf"] = pre / cf.replace(0, np.nan)
    out["free_cf_over_cf"] = cf_free / cf.replace(0, np.nan)
    out["dosage_over_conc"] = dosage / conc.replace(0, np.nan)
    out["volume_over_stimpay"] = vol / stim.replace(0, np.nan)
    out["a_over_a_hat"] = a / a_hat.replace(0, np.nan)
    for c in RATIO_NUM:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    # Group-mean deviations (leave-in; small-n so keep simple)
    def _dev(series: pd.Series, by: pd.Series) -> pd.Series:
        return series - series.groupby(by).transform("mean")

    out["dosage_dev_field"] = _dev(dosage, out["Field"])
    out["conc_dev_field"] = _dev(conc, out["Field"])
    out["volume_dev_field"] = _dev(vol, out["Field"])
    out["dosage_dev_reservoir"] = _dev(dosage, out["Reservoir"])
    out["log_cf_dev_field"] = _dev(log_cf, out["Field"])
    out["log_pre_dev_field"] = _dev(log_pre, out["Field"])

    def _z(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sd = series.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return series * 0.0
        return (series - mu) / sd

    out["dosage_z_global"] = _z(dosage)
    out["log_cf_z_global"] = _z(log_cf)

    return out.sort_values("ID").reset_index(drop=True)


def feature_sets(
    *, target: str = "relative"
) -> dict[str, tuple[list[str], list[str]]]:
    """name -> (categorical cols, numeric cols).

    Always keeps oil_0_180_no_shutdown available in oil / engineered sets.
    For volume_norm, also keeps Total Acid Volume in oil/job-inclusive sets
    and adds CF / volume ablations.
    """
    cf_all = CF_RIDGE_NUM + CF_FREE_NUM
    vol_num = [VOL_COL]
    # For volume_norm: oil context sets carry both CF and acid volume.
    oil_core = (
        PRE_OIL_NUM + cf_all + vol_num
        if target == "volume_norm"
        else PRE_OIL_NUM + cf_all
    )
    well = (WELL_CAT, [])
    well_x = (WELL_CAT + WELL_INTERACTION + SEQ_CAT, SEQ_NUM)
    job = (JOB_CAT, JOB_NUM)
    job_design = ([], JOB_NUM)
    decline = ([], DECLINE_NUM)
    pre_oil = ([], PRE_OIL_NUM)
    cf_oil = ([], cf_all)
    oil_level = ([], oil_core)
    well_job = (WELL_CAT + JOB_CAT, JOB_NUM)
    well_oil = (WELL_CAT, oil_core)
    job_oil = (JOB_CAT, JOB_NUM + PRE_OIL_NUM + cf_all)
    # job_oil already has volume via JOB_NUM; ensure uniqueness
    if target == "volume_norm":
        job_oil = (JOB_CAT, list(dict.fromkeys(JOB_NUM + oil_core)))
    decline_oil = ([], DECLINE_NUM + oil_core)
    well_job_oil = (
        WELL_CAT + JOB_CAT,
        list(dict.fromkeys(JOB_NUM + oil_core)),
    )
    products = ([], PRODUCT_NUM)
    ratios = ([], RATIO_NUM)
    deviations = ([], DEVIATION_NUM)
    engineered = (
        ENGINEERED_CAT,
        PRODUCT_NUM + RATIO_NUM + DEVIATION_NUM,
    )
    # Design × productivity interactions + CF level (+ volume already listed)
    design_x_cf = (
        [],
        [
            "Acid Concentration %",
            "Dosage (Gal/ft)",
            "Total Acid Volume",
            *CF_RIDGE_NUM,
            "conc_x_dosage",
            "dosage_x_log_cf",
            "conc_x_log_cf",
            "dosage_dev_field",
            "conc_dev_field",
            "pre_over_cf",
        ],
    )
    ops_nums = [
        "Acid Concentration %",
        "Dosage (Gal/ft)",
        "oil_neg180_0_observed",
        "oil_0_180_no_shutdown",
        "ridge_b_oof",
        "n",
        "confidence",
        "conc_x_dosage",
        "dosage_x_log_cf",
        "dosage_dev_field",
        "pre_over_cf",
    ]
    if target == "volume_norm":
        # Hard rule: keep volume alongside CF in ops.
        ops_nums = [
            "Acid Concentration %",
            "Dosage (Gal/ft)",
            VOL_COL,
            "oil_neg180_0_observed",
            "oil_0_180_no_shutdown",
            "ridge_b_oof",
            "n",
            "confidence",
            "conc_x_dosage",
            "dosage_x_log_cf",
            "dosage_dev_field",
            "pre_over_cf",
        ]
    ops = (WELL_CAT + JOB_CAT, ops_nums)
    all_feats = (
        WELL_CAT
        + WELL_INTERACTION
        + SEQ_CAT
        + JOB_CAT
        + ENGINEERED_CAT,
        JOB_NUM
        + DECLINE_NUM
        + PRE_OIL_NUM
        + cf_all
        + SEQ_NUM
        + LOG_NUM
        + PRODUCT_NUM
        + RATIO_NUM
        + DEVIATION_NUM,
    )
    sets = {
        "well": well,
        "well_x": well_x,
        "job": job,
        "job_design": job_design,
        "decline": decline,
        "pre_oil": pre_oil,
        "cf_oil": cf_oil,
        "oil_level": oil_level,
        "well+job": well_job,
        "well+oil": well_oil,
        "job+oil": job_oil,
        "decline+oil": decline_oil,
        "well+job+oil": well_job_oil,
        "products": products,
        "ratios": ratios,
        "deviations": deviations,
        "engineered": engineered,
        "design_x_cf": design_x_cf,
        "ops": ops,
        "all": all_feats,
    }
    if target == "volume_norm":
        # Ablations: isolate CF / volume pieces of the target definition.
        sets = {
            "cf_only": ([], CF_RIDGE_NUM),
            "volume_only": ([], vol_num),
            "cf+volume": ([], CF_RIDGE_NUM + vol_num),
            **sets,
        }
    return sets


def make_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if not transformers:
        raise ValueError("Feature set has no columns")
    return ColumnTransformer(transformers)


def make_tree_preprocessor(
    cat_cols: list[str], num_cols: list[str]
) -> ColumnTransformer:
    """One-hot cats + median-impute nums (no scaling) for tree models."""
    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                num_cols,
            )
        )
    return ColumnTransformer(transformers)


def model_specs() -> dict[str, tuple[str, BaseEstimator]]:
    """name -> (preprocessor_kind, estimator).

    preprocessor_kind: 'scaled' | 'tree'
    """
    return {
        "mean": ("scaled", DummyRegressor(strategy="mean")),
        "ols": ("scaled", LinearRegression()),
        "ridge": ("scaled", RidgeCV(alphas=RIDGE_ALPHAS)),
        "lasso": (
            "scaled",
            LassoCV(
                alphas=LASSO_ALPHAS,
                max_iter=8000,
                tol=1e-3,
                cv=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "elasticnet": (
            "scaled",
            ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=8000,
                tol=1e-3,
                random_state=RANDOM_STATE,
            ),
        ),
        "knn5": (
            "scaled",
            KNeighborsRegressor(n_neighbors=5, weights="distance"),
        ),
        "knn11": (
            "scaled",
            KNeighborsRegressor(n_neighbors=11, weights="distance"),
        ),
        "svr_rbf": ("scaled", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
        "rf": (
            "tree",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "gbr": (
            "tree",
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ),
        "hgb": (
            "tree",
            HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def build_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    kind: str,
    estimator: BaseEstimator,
    *,
    y_transform: str,
) -> Pipeline:
    if kind == "scaled":
        pre = make_preprocessor(cat_cols, num_cols)
    elif kind == "tree":
        pre = make_tree_preprocessor(cat_cols, num_cols)
    else:
        raise ValueError(kind)
    if y_transform == "log":
        model: BaseEstimator = LogTargetRegressor(estimator)
    elif y_transform == "asinh":
        model = AsinhTargetRegressor(estimator)
    elif y_transform == "none":
        model = estimator
    else:
        raise ValueError(y_transform)
    return Pipeline([("pre", pre), ("model", model)])


def metrics_dict(y: np.ndarray, yhat: np.ndarray, prefix: str) -> dict:
    return {
        f"r2_{prefix}": float(r2_score(y, yhat)),
        f"mae_{prefix}": float(mean_absolute_error(y, yhat)),
        f"rmse_{prefix}": float(np.sqrt(mean_squared_error(y, yhat))),
        f"medae_{prefix}": float(np.median(np.abs(y - yhat))),
    }


def oof_predict(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
) -> tuple[np.ndarray, int]:
    n_groups = int(groups.nunique())
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError("Not enough well groups for GroupKFold")
    gkf = GroupKFold(n_splits=n_splits)
    yhat = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold = clone(pipe)
        fold.fit(X.iloc[train_idx], y[train_idx])
        yhat[test_idx] = fold.predict(X.iloc[test_idx])
    return yhat, n_splits


def eval_combo(
    name: str,
    feature_set: str,
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    *,
    y_transform: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    pipe_in = clone(pipe)
    pipe_in.fit(X, y)
    yhat_in = pipe_in.predict(X)
    yhat_oof, n_splits = oof_predict(pipe, X, y, groups)

    row = {
        "feature_set": feature_set,
        "model": name,
        "y_transform": y_transform,
        "n": int(len(y)),
        "n_wells": int(groups.nunique()),
        "n_splits": int(n_splits),
        "n_features_raw": int(X.shape[1]),
        "y_mean": float(y.mean()),
        "y_std": float(y.std(ddof=0)),
    }
    row.update(metrics_dict(y, yhat_in, "in_sample"))
    row.update(metrics_dict(y, yhat_oof, "oof"))
    return row, yhat_in, yhat_oof


def prepare_frame(target: str) -> tuple[pd.DataFrame, str, Path]:
    merged = engineer_features(load_merged())
    if target == "absolute":
        out_dir = ROOT / "outputs" / "observed_oil_regression"
        y_col = OBS_COL
    elif target == "relative":
        out_dir = ROOT / "outputs" / "relative_uplift_regression"
        cf = merged[CF_COL].to_numpy(dtype=float)
        obs = merged[OBS_COL].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (obs - cf) / cf
        merged[REL_TARGET] = rel
        ok = np.isfinite(rel) & (cf > 0)
        n_drop = int((~ok).sum())
        if n_drop:
            print(f"Dropping {n_drop} rows with non-finite relative uplift or cf<=0")
        merged = merged.loc[ok].reset_index(drop=True)
        y_col = REL_TARGET
    elif target == "volume_norm":
        out_dir = ROOT / "outputs" / "uplift_per_acid_volume_regression"
        cf = merged[CF_COL].to_numpy(dtype=float)
        obs = merged[OBS_COL].to_numpy(dtype=float)
        vol = merged[VOL_COL].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = (obs - cf) / vol
        merged[VOL_TARGET] = y
        ok = (
            np.isfinite(y)
            & np.isfinite(obs)
            & np.isfinite(cf)
            & np.isfinite(vol)
            & (vol > 0)
        )
        n_drop = int((~ok).sum())
        n_vol_bad = int((~(np.isfinite(vol) & (vol > 0))).sum())
        n_oil_bad = int((~(np.isfinite(obs) & np.isfinite(cf))).sum())
        if n_drop:
            print(
                f"Dropping {n_drop} rows "
                f"(volume<=0 or non-finite: {n_vol_bad}; "
                f"non-finite obs/cf: {n_oil_bad})"
            )
        merged = merged.loc[ok].reset_index(drop=True)
        y_col = VOL_TARGET
    else:
        raise ValueError(target)
    return merged, y_col, out_dir


def y_transforms_for(target: str) -> tuple[str, ...]:
    if target == "absolute":
        return ("none", "log")
    # Relative / volume-norm uplift can be negative — log1p invalid; asinh ok.
    return ("none", "asinh")


def summarize_y(y: np.ndarray, label: str) -> None:
    y = np.asarray(y, dtype=float)
    pct_neg = 100.0 * float((y < 0).mean())
    q = np.quantile(y, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(
        f"{label}: n={len(y)}  mean={y.mean():.4g}  median={np.median(y):.4g}  "
        f"std={y.std(ddof=0):.4g}  %neg={pct_neg:.1f}"
    )
    print(
        f"  tails: min={y.min():.4g}  p1={q[0]:.4g}  p5={q[1]:.4g}  "
        f"p95={q[5]:.4g}  p99={q[6]:.4g}  max={y.max():.4g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("absolute", "relative", "volume_norm"),
        default="relative",
        help=(
            "absolute post oil; (obs-cf)/cf relative uplift (default); "
            "or (obs-cf)/Total Acid Volume"
        ),
    )
    args = parser.parse_args()

    merged, y_col, out_dir = prepare_frame(args.target)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_path = out_dir / "merged_modeling_table.csv"
    merged.to_csv(merged_path, index=False)
    print(
        f"Target={args.target} ({y_col}): n={len(merged)} jobs, "
        f"{merged['Well'].nunique()} wells -> {merged_path}"
    )
    summarize_y(merged[y_col].to_numpy(dtype=float), "y")

    y = merged[y_col].to_numpy(dtype=float)
    groups = merged["Well"]
    sets = feature_sets(target=args.target)
    models = model_specs()
    transforms = y_transforms_for(args.target)

    metrics_rows: list[dict] = []
    id_cols = ["ID", "Well", "Index", y_col]
    if args.target == "relative":
        id_cols = ["ID", "Well", "Index", OBS_COL, CF_COL, y_col]
    elif args.target == "volume_norm":
        id_cols = ["ID", "Well", "Index", OBS_COL, CF_COL, VOL_COL, y_col]
    oof_wide = merged[id_cols].copy()

    for fs_name, (cat_cols, num_cols) in sets.items():
        cols = cat_cols + num_cols
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            raise KeyError(f"Feature set {fs_name} missing columns: {missing}")
        X = merged[cols].copy()
        for y_transform in transforms:
            tag = y_transform
            for model_name, (kind, estimator) in models.items():
                pipe = build_pipeline(
                    cat_cols,
                    num_cols,
                    kind,
                    estimator,
                    y_transform=y_transform,
                )
                combo = f"{fs_name}__{model_name}__{tag}"
                try:
                    row, yhat_in, yhat_oof = eval_combo(
                        model_name,
                        fs_name,
                        pipe,
                        X,
                        y,
                        groups,
                        y_transform=y_transform,
                    )
                except Exception as exc:  # noqa: BLE001 — sweep must continue
                    print(f"SKIP {combo}: {exc}")
                    continue
                metrics_rows.append(row)
                oof_wide[f"oof__{combo}"] = yhat_oof
                print(
                    f"{combo:45s}  "
                    f"OOF R²={row['r2_oof']:+.3f}  "
                    f"MAE={row['mae_oof']:.4g}  "
                    f"RMSE={row['rmse_oof']:.4g}"
                )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["r2_oof", "mae_oof"], ascending=[False, True]
    )
    metrics.to_csv(out_dir / "metrics_all.csv", index=False)

    best_per_fs = (
        metrics.sort_values("r2_oof", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("r2_oof", ascending=False)
    )
    best_per_fs.to_csv(out_dir / "metrics_best_per_feature_set.csv", index=False)

    best = metrics.iloc[0]
    best_key = f"{best['feature_set']}__{best['model']}__{best['y_transform']}"
    summary = {
        "target": args.target,
        "y_col": y_col,
        "best_combo": best_key,
        "feature_set": best["feature_set"],
        "model": best["model"],
        "y_transform": best["y_transform"],
        "r2_oof": float(best["r2_oof"]),
        "mae_oof": float(best["mae_oof"]),
        "rmse_oof": float(best["rmse_oof"]),
        "medae_oof": float(best["medae_oof"]),
        "r2_in_sample": float(best["r2_in_sample"]),
        "overfit_gap_r2": float(best["r2_in_sample"] - best["r2_oof"]),
        "n": int(best["n"]),
        "n_wells": int(best["n_wells"]),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "best_summary.csv", index=False)

    for transform, label in (
        ("none", "raw"),
        ("log", "logtarget"),
        ("asinh", "asinhtarget"),
    ):
        sub = metrics.loc[metrics["y_transform"] == transform]
        if sub.empty:
            continue
        sub.pivot_table(
            index="feature_set", columns="model", values="r2_oof"
        ).to_csv(out_dir / f"pivot_r2_oof_{label}.csv")
        sub.pivot_table(
            index="feature_set", columns="model", values="mae_oof"
        ).to_csv(out_dir / f"pivot_mae_oof_{label}.csv")

    oof_wide.to_csv(out_dir / "oof_predictions_all.csv", index=False)

    slim = merged[id_cols].copy()
    slim["yhat_oof"] = oof_wide[f"oof__{best_key}"]
    slim["residual"] = slim[y_col] - slim["yhat_oof"]
    slim.to_csv(out_dir / "oof_predictions_best.csv", index=False)

    mean_row = metrics.loc[
        (metrics["model"] == "mean") & (metrics["y_transform"] == "none")
    ].iloc[0]
    print("\n=== Top 15 by OOF R² ===")
    cols = [
        "feature_set",
        "model",
        "y_transform",
        "r2_oof",
        "mae_oof",
        "rmse_oof",
        "medae_oof",
        "r2_in_sample",
    ]
    print(metrics[cols].head(15).to_string(index=False))
    print(f"\nBest: {best_key}")
    print(
        f"  OOF R²={summary['r2_oof']:.3f}  "
        f"MAE={summary['mae_oof']:.4g}  RMSE={summary['rmse_oof']:.4g}  "
        f"overfit gap (in−OOF R²)={summary['overfit_gap_r2']:.3f}"
    )
    print(
        f"Mean baseline MAE={mean_row['mae_oof']:.4g}  "
        f"(best MAE / mean = {summary['mae_oof'] / mean_row['mae_oof']:.3f})"
    )

    if args.target == "volume_norm":
        design_sets = {"job", "job_design", "well+job"}
        oil_sets = {
            "cf_only",
            "volume_only",
            "cf+volume",
            "cf_oil",
            "oil_level",
            "well+oil",
            "decline+oil",
        }
        best_design = best_per_fs.loc[
            best_per_fs["feature_set"].isin(design_sets), "r2_oof"
        ]
        best_oil = best_per_fs.loc[
            best_per_fs["feature_set"].isin(oil_sets), "r2_oof"
        ]
        print("\n=== Design vs oil/volume context ===")
        if len(best_design):
            print(
                f"Best design-family OOF R²={best_design.max():+.3f} "
                f"({best_per_fs.loc[best_design.idxmax(), 'feature_set']})"
            )
        if len(best_oil):
            print(
                f"Best oil/vol-context OOF R²={best_oil.max():+.3f} "
                f"({best_per_fs.loc[best_oil.idxmax(), 'feature_set']})"
            )
        design_helps = (
            len(best_design)
            and len(best_oil)
            and float(best_design.max()) > float(best_oil.max()) + 0.02
        )
        print(
            "Design knobs improve over oil/volume context: "
            + ("yes" if design_helps else "no")
        )

    print(f"Wrote artifacts under {out_dir}")


if __name__ == "__main__":
    main()