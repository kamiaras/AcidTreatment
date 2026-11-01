"""Fuzzy ternary classification of relative oil uplift (NB / ZE / PB).

Joins (inner only; same Well zip-join as regress_observed_oil.py):

  - data/processed/job_parameters.xlsx
  - data/processed/well_characteristics.xlsx
  - outputs/well_timeseries/exp_decline_fits.csv
  - outputs/well_timeseries/observed_oil_0_180.csv

Target class from relative uplift

  r = (oil_0_180_observed - oil_0_180_no_shutdown) / oil_0_180_no_shutdown

Hard labels (symmetric fixed hedges around zero):

  ZE  iff  |r| <= eps
  NB  iff  r < -eps
  PB  iff  r >  eps

Default eps = 0.20 (±20% indifference band). A median(|r|) option is
also available; the class balance depends on the supplied data.

Also stores triangular fuzzy memberships mu_NB, mu_ZE, mu_PB for EDA;
supervised task is hard multiclass {NB, ZE, PB}.

Never uses oil_0_180_observed, r, or F1–F5 as features. May keep
oil_0_180_no_shutdown as a feature.

CV: GroupKFold by Well. Primary metrics: OOF macro-F1, balanced accuracy.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sys

from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Reuse join + feature engineering from the regression predecessor.
from regress_observed_oil import (  # noqa: E402
    CF_COL,
    OBS_COL,
    engineer_features,
    feature_sets,
    load_merged,
    make_preprocessor,
    make_tree_preprocessor,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "relative_uplift_fuzzy"

N_SPLITS = 5
RANDOM_STATE = 0
REL_COL = "relative_uplift"
LABEL_COL = "uplift_class"
CLASS_ORDER = ["NB", "ZE", "PB"]


def compute_relative_uplift(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cf = out[CF_COL].to_numpy(dtype=float)
    obs = out[OBS_COL].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (obs - cf) / cf
    out[REL_COL] = r
    ok = np.isfinite(r) & (cf > 0)
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"Dropping {n_drop} rows with non-finite r or cf<=0")
    return out.loc[ok].reset_index(drop=True)


def choose_eps(r: np.ndarray, method: str) -> float:
    abs_r = np.abs(r)
    if method == "median_abs":
        eps = float(np.median(abs_r))
    elif method == "p25_abs":
        eps = float(np.quantile(abs_r, 0.25))
    elif method == "p33_abs":
        eps = float(np.quantile(abs_r, 1.0 / 3.0))
    elif method == "fixed_0_2":
        eps = 0.20
    elif method == "fixed_0_25":
        eps = 0.25
    else:
        raise ValueError(f"Unknown eps method: {method}")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"Invalid eps={eps} from method={method}")
    return eps


def assign_hard_labels(r: np.ndarray, eps: float) -> np.ndarray:
    labels = np.full(len(r), "ZE", dtype=object)
    labels[r < -eps] = "NB"
    labels[r > eps] = "PB"
    return labels


def triangular_memberships(r: np.ndarray, eps: float) -> pd.DataFrame:
    """Triangular / shoulder memberships on r with indifference half-width eps.

    Hard-label boundaries at ±eps; soft shoulders over [±eps, ±2eps]:

      mu_NB = 1 for r <= -2eps; linear 1→0 on [-2eps, -eps]; 0 for r >= -eps
      mu_PB = 1 for r >= +2eps; linear 0→1 on [+eps, +2eps]; 0 for r <= +eps
      mu_ZE = triangular peak 1 at 0, support [-2eps, +2eps]

    Soft grades are L1-normalized to sum to 1 (EDA only; training uses hard labels).
    """
    r = np.asarray(r, dtype=float)
    half = float(eps)
    outer = 2.0 * half

    mu_nb = np.zeros_like(r)
    mu_nb[r <= -outer] = 1.0
    mid_nb = (r > -outer) & (r < -half)
    mu_nb[mid_nb] = (-half - r[mid_nb]) / half

    mu_pb = np.zeros_like(r)
    mu_pb[r >= outer] = 1.0
    mid_pb = (r > half) & (r < outer)
    mu_pb[mid_pb] = (r[mid_pb] - half) / half

    mu_ze = np.zeros_like(r)
    left_ze = (r >= -outer) & (r <= 0)
    right_ze = (r > 0) & (r <= outer)
    mu_ze[left_ze] = (r[left_ze] + outer) / outer
    mu_ze[right_ze] = (outer - r[right_ze]) / outer

    raw = np.column_stack([mu_nb, mu_ze, mu_pb])
    s = raw.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    norm = raw / s
    return pd.DataFrame(
        {
            "mu_NB": norm[:, 0],
            "mu_ZE": norm[:, 1],
            "mu_PB": norm[:, 2],
        }
    )


def class_definition_summary(df: pd.DataFrame, eps: float, method: str) -> dict:
    r = df[REL_COL].to_numpy(dtype=float)
    y = df[LABEL_COL]
    counts = y.value_counts().reindex(CLASS_ORDER, fill_value=0)
    ranges = {}
    for lab in CLASS_ORDER:
        rr = r[y.to_numpy() == lab]
        if len(rr) == 0:
            ranges[lab] = {"n": 0, "r_min": None, "r_max": None, "r_mean": None}
        else:
            ranges[lab] = {
                "n": int(len(rr)),
                "r_min": float(rr.min()),
                "r_max": float(rr.max()),
                "r_mean": float(rr.mean()),
                "r_median": float(np.median(rr)),
            }
    return {
        "eps_method": method,
        "eps": float(eps),
        "rule": "ZE iff |r|<=eps; NB iff r<-eps; PB iff r>eps",
        "n": int(len(df)),
        "n_wells": int(df["Well"].nunique()),
        "r_mean": float(r.mean()),
        "r_median": float(np.median(r)),
        "r_std": float(r.std(ddof=0)),
        "abs_r_median": float(np.median(np.abs(r))),
        "frac_negative_r": float((r < 0).mean()),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_fractions": {
            k: float(v) / len(df) for k, v in counts.items()
        },
        "class_r_ranges": ranges,
        "imbalance_ratio_max_min": float(
            counts.max() / max(counts.min(), 1)
        ),
    }


def model_specs() -> dict[str, tuple[str, BaseEstimator]]:
    """name -> (preprocessor_kind, estimator)."""
    return {
        "majority": (
            "scaled",
            DummyClassifier(strategy="most_frequent"),
        ),
        "stratified": (
            "scaled",
            DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        ),
        "logreg": (
            "scaled",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "logreg_C05": (
            "scaled",
            LogisticRegression(
                C=0.5,
                solver="lbfgs",
                max_iter=4000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "ridge": (
            "scaled",
            RidgeClassifier(alpha=1.0, class_weight="balanced"),
        ),
        "knn5": (
            "scaled",
            KNeighborsClassifier(n_neighbors=5, weights="distance"),
        ),
        "knn11": (
            "scaled",
            KNeighborsClassifier(n_neighbors=11, weights="distance"),
        ),
        "svc_rbf": (
            "scaled",
            SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "rf": (
            "tree",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "hgb": (
            "tree",
            HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.05,
                max_iter=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    }


def build_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
    kind: str,
    estimator: BaseEstimator,
) -> Pipeline:
    if kind == "scaled":
        pre = make_preprocessor(cat_cols, num_cols)
    elif kind == "tree":
        pre = make_tree_preprocessor(cat_cols, num_cols)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("model", estimator)])


def oof_predict_labels(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    le: LabelEncoder,
) -> tuple[np.ndarray, int]:
    n_groups = int(groups.nunique())
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError("Not enough well groups for GroupKFold")
    gkf = GroupKFold(n_splits=n_splits)
    yhat = np.full(len(y), -1, dtype=int)
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold = clone(pipe)
        # Stratified dummy needs y as array; class_weight models need all classes
        # present in train when possible — GroupKFold may miss a rare class.
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            # Degenerate fold: predict majority of train
            maj = int(np.bincount(y_tr).argmax())
            yhat[test_idx] = maj
            continue
        try:
            fold.fit(X.iloc[train_idx], y_tr)
            pred = fold.predict(X.iloc[test_idx])
            yhat[test_idx] = pred
        except Exception:
            # Rare class may be absent from a GroupKFold train split.
            maj = int(np.bincount(y_tr).argmax())
            yhat[test_idx] = maj
    if (yhat < 0).any():
        raise RuntimeError("Incomplete OOF predictions")
    return yhat, n_splits


def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> dict:
    labels_idx = list(range(len(le.classes_)))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=labels_idx, zero_division=0)
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0
    )
    out: dict = {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
    }
    for i, name in enumerate(le.classes_):
        out[f"precision_{name}"] = float(p[i])
        out[f"recall_{name}"] = float(r[i])
        out[f"f1_{name}"] = float(f[i])
        out[f"support_{name}"] = int(s[i])
    return out


def nb_pb_binary_metrics(
    y_true_lab: np.ndarray, y_pred_lab: np.ndarray
) -> dict:
    """Restrict to true NB/PB rows; treat as binary NB vs PB (ZE preds = miss)."""
    mask = np.isin(y_true_lab, ["NB", "PB"])
    if mask.sum() == 0:
        return {
            "n_nb_pb": 0,
            "nb_pb_balanced_accuracy": float("nan"),
            "nb_pb_macro_f1": float("nan"),
            "nb_pb_accuracy_strict": float("nan"),
            "frac_confused_nb_pb": float("nan"),
        }
    yt = y_true_lab[mask]
    yp = y_pred_lab[mask]
    # Strict: ZE prediction counts as wrong for both
    strict_acc = float((yt == yp).mean())
    # Map ZE predictions to wrong class for balanced metrics via force
    yp_bin = yp.copy()
    # For F1 on {NB,PB}, map non-{NB,PB} preds to the opposite of truth (always wrong)
    for i in range(len(yp_bin)):
        if yp_bin[i] not in ("NB", "PB"):
            yp_bin[i] = "PB" if yt[i] == "NB" else "NB"
    bal = float(balanced_accuracy_score(yt, yp_bin))
    mac = float(f1_score(yt, yp_bin, average="macro", labels=["NB", "PB"], zero_division=0))
    # Direct NB↔PB swaps among rows predicted as NB or PB
    both = np.isin(yp, ["NB", "PB"])
    if both.sum():
        confused = float(((yt[both] == "NB") & (yp[both] == "PB")).sum()
                         + ((yt[both] == "PB") & (yp[both] == "NB")).sum()) / float(mask.sum())
    else:
        confused = float("nan")
    return {
        "n_nb_pb": int(mask.sum()),
        "nb_pb_balanced_accuracy": bal,
        "nb_pb_macro_f1": mac,
        "nb_pb_accuracy_strict": strict_acc,
        "frac_confused_nb_pb": confused,
    }


def eval_combo(
    name: str,
    feature_set: str,
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    y_lab: np.ndarray,
    groups: pd.Series,
    le: LabelEncoder,
) -> tuple[dict, np.ndarray, np.ndarray]:
    pipe_in = clone(pipe)
    pipe_in.fit(X, y)
    yhat_in = pipe_in.predict(X)
    yhat_oof, n_splits = oof_predict_labels(pipe, X, y, groups, le)

    row = {
        "feature_set": feature_set,
        "model": name,
        "n": int(len(y)),
        "n_wells": int(groups.nunique()),
        "n_splits": int(n_splits),
        "n_features_raw": int(X.shape[1]),
    }
    row.update({f"{k}_in_sample": v for k, v in cls_metrics(y, yhat_in, le).items()})
    oof_m = cls_metrics(y, yhat_oof, le)
    row.update({f"{k}_oof": v for k, v in oof_m.items()})

    yhat_oof_lab = le.inverse_transform(yhat_oof)
    row.update({f"{k}_oof": v for k, v in nb_pb_binary_metrics(y_lab, yhat_oof_lab).items()})
    return row, yhat_in, yhat_oof


def majority_baseline_metrics(y: np.ndarray, le: LabelEncoder) -> dict:
    maj = int(np.bincount(y).argmax())
    yhat = np.full_like(y, maj)
    m = cls_metrics(y, yhat, le)
    return {
        "majority_class": str(le.inverse_transform([maj])[0]),
        **{f"majority_{k}": v for k, v in m.items()},
    }


def write_confusion(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    le: LabelEncoder,
) -> None:
    labels = list(range(len(le.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in le.classes_],
        columns=[f"pred_{c}" for c in le.classes_],
    )
    cm_df.to_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eps-method",
        choices=("fixed_0_2", "fixed_0_25", "p25_abs", "p33_abs", "median_abs"),
        default="fixed_0_2",
        help="ZE indifference half-width eps (default: fixed 0.20)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = engineer_features(compute_relative_uplift(load_merged()))
    r = merged[REL_COL].to_numpy(dtype=float)
    eps = choose_eps(r, args.eps_method)
    merged[LABEL_COL] = assign_hard_labels(r, eps)
    mus = triangular_memberships(r, eps)
    merged = pd.concat([merged, mus], axis=1)

    cdef = class_definition_summary(merged, eps, args.eps_method)
    (out_dir / "class_definition.json").write_text(
        json.dumps(cdef, indent=2) + "\n"
    )
    pd.DataFrame(
        [
            {
                "label": lab,
                **cdef["class_r_ranges"][lab],
                "fraction": cdef["class_fractions"][lab],
            }
            for lab in CLASS_ORDER
        ]
    ).to_csv(out_dir / "class_definition_summary.csv", index=False)

    print(
        f"n={cdef['n']} jobs / {cdef['n_wells']} wells | "
        f"eps={eps:.4g} ({args.eps_method})"
    )
    print(
        "Class counts: "
        + ", ".join(f"{k}={v}" for k, v in cdef["class_counts"].items())
        + f" | imbalance max/min={cdef['imbalance_ratio_max_min']:.2f}"
    )
    print(
        f"r: mean={cdef['r_mean']:.3g}  median={cdef['r_median']:.3g}  "
        f"frac_neg={cdef['frac_negative_r']:.1%}"
    )

    merged_path = out_dir / "merged_modeling_table.csv"
    merged.to_csv(merged_path, index=False)

    le = LabelEncoder()
    # Force CLASS_ORDER encoding
    le.fit(CLASS_ORDER)
    y = le.transform(merged[LABEL_COL].to_numpy())
    y_lab = merged[LABEL_COL].to_numpy()
    groups = merged["Well"]

    baseline = majority_baseline_metrics(y, le)
    pd.DataFrame([baseline]).to_csv(out_dir / "majority_baseline.csv", index=False)
    print(
        f"Majority baseline ({baseline['majority_class']}): "
        f"macro-F1={baseline['majority_macro_f1']:.3f}  "
        f"bal-acc={baseline['majority_balanced_accuracy']:.3f}"
    )

    sets = feature_sets()
    models = model_specs()
    metrics_rows: list[dict] = []
    id_cols = [
        "ID",
        "Well",
        "Index",
        OBS_COL,
        CF_COL,
        REL_COL,
        LABEL_COL,
        "mu_NB",
        "mu_ZE",
        "mu_PB",
    ]
    oof_wide = merged[id_cols].copy()

    for fs_name, (cat_cols, num_cols) in sets.items():
        cols = cat_cols + num_cols
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            raise KeyError(f"Feature set {fs_name} missing columns: {missing}")
        # Leakage guards
        for banned in (OBS_COL, REL_COL, LABEL_COL, "mu_NB", "mu_ZE", "mu_PB"):
            if banned in cols:
                raise RuntimeError(f"Leakage: {banned} in feature set {fs_name}")
        X = merged[cols].copy()
        for model_name, (kind, estimator) in models.items():
            pipe = build_pipeline(cat_cols, num_cols, kind, estimator)
            combo = f"{fs_name}__{model_name}"
            try:
                row, yhat_in, yhat_oof = eval_combo(
                    model_name,
                    fs_name,
                    pipe,
                    X,
                    y,
                    y_lab,
                    groups,
                    le,
                )
            except Exception as exc:  # noqa: BLE001 — sweep must continue
                print(f"SKIP {combo}: {exc}")
                continue
            metrics_rows.append(row)
            oof_lab = le.inverse_transform(yhat_oof)
            oof_wide[f"oof__{combo}"] = oof_lab
            print(
                f"{combo:40s}  "
                f"macroF1={row['macro_f1_oof']:.3f}  "
                f"balAcc={row['balanced_accuracy_oof']:.3f}  "
                f"NB↔PB_bal={row['nb_pb_balanced_accuracy_oof']:.3f}"
            )

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["macro_f1_oof", "balanced_accuracy_oof"],
        ascending=[False, False],
    )
    metrics.to_csv(out_dir / "metrics_all.csv", index=False)

    best_per_fs = (
        metrics.sort_values(
            ["macro_f1_oof", "balanced_accuracy_oof"],
            ascending=[False, False],
        )
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("macro_f1_oof", ascending=False)
    )
    best_per_fs.to_csv(out_dir / "metrics_best_per_feature_set.csv", index=False)

    best = metrics.iloc[0]
    best_key = f"{best['feature_set']}__{best['model']}"
    summary = {
        "eps_method": args.eps_method,
        "eps": float(eps),
        "best_combo": best_key,
        "feature_set": best["feature_set"],
        "model": best["model"],
        "macro_f1_oof": float(best["macro_f1_oof"]),
        "balanced_accuracy_oof": float(best["balanced_accuracy_oof"]),
        "accuracy_oof": float(best["accuracy_oof"]),
        "nb_pb_balanced_accuracy_oof": float(best["nb_pb_balanced_accuracy_oof"]),
        "nb_pb_macro_f1_oof": float(best["nb_pb_macro_f1_oof"]),
        "nb_pb_accuracy_strict_oof": float(best["nb_pb_accuracy_strict_oof"]),
        "majority_macro_f1": float(baseline["majority_macro_f1"]),
        "majority_balanced_accuracy": float(baseline["majority_balanced_accuracy"]),
        "n": int(best["n"]),
        "n_wells": int(best["n_wells"]),
        **{f"class_count_{k}": v for k, v in cdef["class_counts"].items()},
    }
    pd.DataFrame([summary]).to_csv(out_dir / "best_summary.csv", index=False)

    metrics.pivot_table(
        index="feature_set", columns="model", values="macro_f1_oof"
    ).to_csv(out_dir / "pivot_macro_f1_oof.csv")
    metrics.pivot_table(
        index="feature_set", columns="model", values="balanced_accuracy_oof"
    ).to_csv(out_dir / "pivot_balanced_accuracy_oof.csv")

    oof_wide.to_csv(out_dir / "oof_predictions_all.csv", index=False)

    # Best OOF confusion + slim preds
    best_pred = oof_wide[f"oof__{best_key}"].to_numpy()
    best_pred_idx = le.transform(best_pred)
    write_confusion(
        out_dir / "confusion_matrix_best_oof.csv",
        y,
        best_pred_idx,
        le,
    )
    # Also write per-feature-set best confusion for design vs oil contrast
    for fs in ("job_design", "oil_level", "well+oil", "well", "all"):
        sub = metrics.loc[metrics["feature_set"] == fs]
        if sub.empty:
            continue
        key = f"{fs}__{sub.iloc[0]['model']}"
        col = f"oof__{key}"
        if col not in oof_wide.columns:
            continue
        pred_idx = le.transform(oof_wide[col].to_numpy())
        write_confusion(
            out_dir / f"confusion_matrix_{fs}_best_oof.csv",
            y,
            pred_idx,
            le,
        )

    slim = merged[id_cols].copy()
    slim["yhat_oof"] = best_pred
    slim["correct"] = slim[LABEL_COL] == slim["yhat_oof"]
    slim.to_csv(out_dir / "oof_predictions_best.csv", index=False)

    # Design vs context contrast table
    contrast_sets = [
        "job_design",
        "job",
        "well",
        "oil_level",
        "cf_oil",
        "well+oil",
        "job+oil",
        "design_x_cf",
        "ops",
        "all",
    ]
    contrast_rows = []
    for fs in contrast_sets:
        sub = metrics.loc[metrics["feature_set"] == fs]
        if sub.empty:
            continue
        top = sub.iloc[0]
        contrast_rows.append(
            {
                "feature_set": fs,
                "best_model": top["model"],
                "macro_f1_oof": top["macro_f1_oof"],
                "balanced_accuracy_oof": top["balanced_accuracy_oof"],
                "nb_pb_balanced_accuracy_oof": top["nb_pb_balanced_accuracy_oof"],
                "nb_pb_macro_f1_oof": top["nb_pb_macro_f1_oof"],
                "recall_NB_oof": top["recall_NB_oof"],
                "recall_PB_oof": top["recall_PB_oof"],
                "recall_ZE_oof": top["recall_ZE_oof"],
            }
        )
    pd.DataFrame(contrast_rows).to_csv(
        out_dir / "design_vs_context_contrast.csv", index=False
    )

    print("\n=== Top 15 by OOF macro-F1 ===")
    cols_show = [
        "feature_set",
        "model",
        "macro_f1_oof",
        "balanced_accuracy_oof",
        "accuracy_oof",
        "nb_pb_balanced_accuracy_oof",
        "recall_NB_oof",
        "recall_ZE_oof",
        "recall_PB_oof",
    ]
    print(metrics[cols_show].head(15).to_string(index=False))

    print(f"\nBest: {best_key}")
    print(
        f"  OOF macro-F1={summary['macro_f1_oof']:.3f}  "
        f"bal-acc={summary['balanced_accuracy_oof']:.3f}  "
        f"(majority macro-F1={summary['majority_macro_f1']:.3f}, "
        f"bal-acc={summary['majority_balanced_accuracy']:.3f})"
    )
    print(
        f"  NB↔PB (true NB/PB only): bal-acc={summary['nb_pb_balanced_accuracy_oof']:.3f}  "
        f"macro-F1={summary['nb_pb_macro_f1_oof']:.3f}  "
        f"strict-acc={summary['nb_pb_accuracy_strict_oof']:.3f}"
    )

    # Separability verdict for console
    nb_pb_bal = summary["nb_pb_balanced_accuracy_oof"]
    design_row = next((r for r in contrast_rows if r["feature_set"] == "job_design"), None)
    oil_row = next((r for r in contrast_rows if r["feature_set"] == "oil_level"), None)
    if nb_pb_bal < 0.55:
        sep = "NB vs PB is NOT clearly separable from available features"
    elif nb_pb_bal < 0.65:
        sep = "NB vs PB shows only weak/marginal separability"
    else:
        sep = (
            "NB vs PB shows modest oil/CF-level separability "
            "(fragile: small NB support; design knobs still null)"
        )
    print(f"  Verdict: {sep}")
    if design_row and oil_row:
        print(
            f"  Design knobs best macro-F1={design_row['macro_f1_oof']:.3f} vs "
            f"oil_level={oil_row['macro_f1_oof']:.3f} "
            f"(NB↔PB bal {design_row['nb_pb_balanced_accuracy_oof']:.3f} vs "
            f"{oil_row['nb_pb_balanced_accuracy_oof']:.3f})"
        )
    print(f"Wrote artifacts under {out_dir}")


if __name__ == "__main__":
    main()
