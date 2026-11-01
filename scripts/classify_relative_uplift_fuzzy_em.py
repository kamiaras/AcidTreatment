"""EM-style learning of NB/ZE/PB boundaries + classifier for relative uplift.

Same joins / features / leakage rules as classify_relative_uplift_fuzzy.py.

Continuous relative uplift

  r = (oil_0_180_observed - oil_0_180_no_shutdown) / oil_0_180_no_shutdown

is partitioned by learned thresholds (theta_L, theta_H) with theta_L < 0 < theta_H:

  NB  iff  r <  theta_L
  ZE  iff  theta_L <= r <= theta_H
  PB  iff  r >  theta_H

Hard-EM / classification-EM (CEM):

  E-step:  z_i = hard label from current (theta_L, theta_H) and r_i
           (soft memberships used for the boundary M-step objective)
  M-step φ: fit multinomial logistic p_φ(z|x) (class-balanced)
  M-step θ: grid-search (theta_L, theta_H) maximizing the expected complete
            log-likelihood under the *fixed* classifier:
              Q(θ) = Σ_i Σ_k μ_k(r_i; θ) log p_φ(k | x_i)
            subject to min class counts and theta_L < 0 < theta_H.

Optional --symmetric forces theta_H = -theta_L = eps (1-D search).

Honest evaluation: outer GroupKFold by Well; EM (boundaries + clf) runs on
train folds only; test labels use train-learned thresholds on test r.

Outputs: outputs/relative_uplift_fuzzy_em/
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from classify_relative_uplift_fuzzy import (  # noqa: E402
    CLASS_ORDER,
    LABEL_COL,
    REL_COL,
    build_pipeline,
    compute_relative_uplift,
    cls_metrics,
    nb_pb_binary_metrics,
)
from regress_observed_oil import (  # noqa: E402
    CF_COL,
    OBS_COL,
    engineer_features,
    feature_sets,
    load_merged,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "relative_uplift_fuzzy_em"

N_SPLITS = 5
RANDOM_STATE = 0
MIN_CLASS_COUNT = 5


def hard_labels(r: np.ndarray, theta_l: float, theta_h: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    lab = np.full(len(r), "ZE", dtype=object)
    lab[r < theta_l] = "NB"
    lab[r > theta_h] = "PB"
    return lab


def soft_memberships(
    r: np.ndarray, theta_l: float, theta_h: float
) -> np.ndarray:
    """Asymmetric triangular / shoulder memberships; rows L1-normalized.

    Shoulders span [theta_L - w_L, theta_L] and [theta_H, theta_H + w_H]
    with w_L = |theta_L|, w_H = theta_H (so outer tips at 2*theta_*).
    """
    r = np.asarray(r, dtype=float)
    w_l = max(abs(theta_l), 1e-6)
    w_h = max(theta_h, 1e-6)
    outer_l = theta_l - w_l
    outer_h = theta_h + w_h

    mu_nb = np.zeros_like(r)
    mu_nb[r <= outer_l] = 1.0
    mid_nb = (r > outer_l) & (r < theta_l)
    mu_nb[mid_nb] = (theta_l - r[mid_nb]) / w_l

    mu_pb = np.zeros_like(r)
    mu_pb[r >= outer_h] = 1.0
    mid_pb = (r > theta_h) & (r < outer_h)
    mu_pb[mid_pb] = (r[mid_pb] - theta_h) / w_h

    mu_ze = np.zeros_like(r)
    # Rise from outer_l → 0 peak region, fall toward outer_h
    # Peak plateau on [theta_l, theta_h]: linear tent from outer_l to mid to outer_h
    mid = 0.5 * (theta_l + theta_h)
    left = (r >= outer_l) & (r <= mid)
    right = (r > mid) & (r <= outer_h)
    denom_l = max(mid - outer_l, 1e-6)
    denom_r = max(outer_h - mid, 1e-6)
    mu_ze[left] = (r[left] - outer_l) / denom_l
    mu_ze[right] = (outer_h - r[right]) / denom_r

    raw = np.column_stack([mu_nb, mu_ze, mu_pb])
    s = raw.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return raw / s


def class_counts_ok(
    r: np.ndarray, theta_l: float, theta_h: float, min_count: int
) -> bool:
    if not (theta_l < 0.0 < theta_h):
        return False
    lab = hard_labels(r, theta_l, theta_h)
    for c in CLASS_ORDER:
        if int((lab == c).sum()) < min_count:
            return False
    return True


def boundary_candidates(
    r: np.ndarray,
    *,
    symmetric: bool,
    min_count: int,
    n_grid: int,
) -> list[tuple[float, float]]:
    """Feasible (theta_L, theta_H) grid with min class counts on this r."""
    r = np.asarray(r, dtype=float)
    neg = r[r < 0]
    pos = r[r > 0]
    if len(neg) < min_count or len(pos) < min_count:
        # Fall back to tiny band around 0 so EM can still run / fail clearly
        return [(-0.05, 0.05)]

    # theta_L in (min_neg, 0): more negative → fewer NB
    t_l_grid = np.unique(
        np.quantile(neg, np.linspace(0.05, 0.95, n_grid))
    )
    t_h_grid = np.unique(
        np.quantile(pos, np.linspace(0.05, 0.95, n_grid))
    )
    # Also include round linguistic seeds
    for seed in (0.1, 0.15, 0.2, 0.25, 0.3):
        if (neg < -seed).sum() >= min_count:
            t_l_grid = np.unique(np.concatenate([t_l_grid, [-seed]]))
        if (pos > seed).sum() >= min_count:
            t_h_grid = np.unique(np.concatenate([t_h_grid, [seed]]))

    cands: list[tuple[float, float]] = []
    if symmetric:
        eps_grid = np.unique(
            np.concatenate(
                [
                    np.abs(t_l_grid),
                    t_h_grid,
                    [0.1, 0.15, 0.2, 0.25, 0.3],
                ]
            )
        )
        for eps in eps_grid:
            if eps <= 0:
                continue
            th = (float(-eps), float(eps))
            if class_counts_ok(r, *th, min_count):
                cands.append(th)
    else:
        for tl in t_l_grid:
            for th in t_h_grid:
                pair = (float(tl), float(th))
                if class_counts_ok(r, *pair, min_count):
                    cands.append(pair)

    if not cands:
        # Loosen once
        return boundary_candidates(
            r, symmetric=symmetric, min_count=max(2, min_count - 2), n_grid=n_grid
        )
    # Deduplicate
    return sorted(set(cands))


def expected_complete_ll(
    proba: np.ndarray, mu: np.ndarray, eps: float = 1e-12
) -> float:
    """Q = mean_i Σ_k μ_ik log p_ik."""
    return float(np.mean(np.sum(mu * np.log(np.clip(proba, eps, 1.0)), axis=1)))


def make_logreg_pipe(cat_cols: list[str], num_cols: list[str], C: float) -> Pipeline:
    est = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return build_pipeline(cat_cols, num_cols, "scaled", est)


def predict_proba_ordered(pipe: Pipeline, X: pd.DataFrame, le: LabelEncoder) -> np.ndarray:
    """Return (n, 3) proba columns in CLASS_ORDER; missing classes → 0."""
    proba_raw = pipe.predict_proba(X)
    classes = list(pipe.named_steps["model"].classes_)
    # classes are integer codes from le
    out = np.zeros((len(X), len(CLASS_ORDER)), dtype=float)
    for j, code in enumerate(classes):
        name = le.inverse_transform([int(code)])[0]
        k = CLASS_ORDER.index(name)
        out[:, k] = proba_raw[:, j]
    # renormalize rows that got mass
    s = out.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return out / s


def fit_classifier(
    pipe: Pipeline,
    X: pd.DataFrame,
    y_lab: np.ndarray,
    le: LabelEncoder,
) -> Pipeline:
    y = le.transform(y_lab)
    # Need ≥2 classes present
    if len(np.unique(y)) < 2:
        raise ValueError("Fewer than 2 classes in train labels")
    fitted = clone(pipe)
    fitted.fit(X, y)
    return fitted


def m_step_boundaries(
    r: np.ndarray,
    proba: np.ndarray,
    candidates: list[tuple[float, float]],
    min_count: int,
) -> tuple[float, float, float]:
    """Pick θ maximizing Q under fixed classifier proba."""
    best = candidates[0]
    best_q = -np.inf
    for th in candidates:
        if not class_counts_ok(r, *th, min_count):
            continue
        mu = soft_memberships(r, *th)
        q = expected_complete_ll(proba, mu)
        if q > best_q:
            best_q = q
            best = th
    return best[0], best[1], float(best_q)


def em_fit(
    X: pd.DataFrame,
    r: np.ndarray,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    symmetric: bool,
    min_count: int,
    n_grid: int,
    max_iter: int,
    C: float,
    tol: float = 1e-4,
) -> dict:
    """Run CEM on one train set. Returns theta, fitted pipe, history, labels."""
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    candidates = boundary_candidates(
        r, symmetric=symmetric, min_count=min_count, n_grid=n_grid
    )

    # Init: prefer linguistic ±0.2 if feasible, else candidate with max min-count balance
    init = (-0.2, 0.2)
    if class_counts_ok(r, *init, min_count) and (
        not symmetric or abs(init[0] + init[1]) < 1e-12
    ):
        theta_l, theta_h = init
    else:
        # Most balanced among candidates (minimize max/min count ratio)
        def _imbalance(th: tuple[float, float]) -> float:
            lab = hard_labels(r, *th)
            cnts = np.array([(lab == c).sum() for c in CLASS_ORDER], dtype=float)
            return float(cnts.max() / max(cnts.min(), 1.0))

        theta_l, theta_h = min(candidates, key=_imbalance)

    pipe0 = make_logreg_pipe(cat_cols, num_cols, C)
    history: list[dict] = []
    pipe_fitted: Pipeline | None = None

    for it in range(max_iter):
        y_lab = hard_labels(r, theta_l, theta_h)
        try:
            pipe_fitted = fit_classifier(pipe0, X, y_lab, le)
        except Exception as exc:  # noqa: BLE001
            history.append(
                {
                    "iter": it,
                    "theta_L": theta_l,
                    "theta_H": theta_h,
                    "status": f"clf_fail:{exc}",
                }
            )
            break

        proba = predict_proba_ordered(pipe_fitted, X, le)
        mu = soft_memberships(r, theta_l, theta_h)
        q_cur = expected_complete_ll(proba, mu)

        new_l, new_h, q_new = m_step_boundaries(
            r, proba, candidates, min_count
        )
        history.append(
            {
                "iter": it,
                "theta_L": theta_l,
                "theta_H": theta_h,
                "Q": q_cur,
                "class_counts": {
                    c: int((y_lab == c).sum()) for c in CLASS_ORDER
                },
                "theta_L_next": new_l,
                "theta_H_next": new_h,
                "Q_next": q_new,
            }
        )

        if abs(new_l - theta_l) < tol and abs(new_h - theta_h) < tol:
            theta_l, theta_h = new_l, new_h
            break
        # Also stop if Q did not improve after move (CEM can oscillate)
        if it > 0 and q_new + 1e-9 < q_cur and abs(new_l - theta_l) + abs(new_h - theta_h) > tol:
            # keep previous θ; already have pipe for previous labels
            break
        theta_l, theta_h = new_l, new_h

    # Final refit at converged θ
    y_lab = hard_labels(r, theta_l, theta_h)
    pipe_fitted = fit_classifier(pipe0, X, y_lab, le)
    return {
        "theta_L": float(theta_l),
        "theta_H": float(theta_h),
        "pipe": pipe_fitted,
        "le": le,
        "labels": y_lab,
        "history": history,
        "n_candidates": len(candidates),
        "class_counts": {c: int((y_lab == c).sum()) for c in CLASS_ORDER},
    }


def oof_em(
    X: pd.DataFrame,
    r: np.ndarray,
    groups: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
    *,
    symmetric: bool,
    min_count: int,
    n_grid: int,
    max_iter: int,
    C: float,
) -> dict:
    n_groups = int(groups.nunique())
    n_splits = min(N_SPLITS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    le = LabelEncoder()
    le.fit(CLASS_ORDER)

    y_true_lab = np.full(len(r), "", dtype=object)
    y_pred_lab = np.full(len(r), "", dtype=object)
    theta_rows: list[dict] = []

    for fold_id, (tr, te) in enumerate(gkf.split(X, groups=groups)):
        fit = em_fit(
            X.iloc[tr],
            r[tr],
            cat_cols,
            num_cols,
            symmetric=symmetric,
            min_count=min_count,
            n_grid=n_grid,
            max_iter=max_iter,
            C=C,
        )
        theta_l = fit["theta_L"]
        theta_h = fit["theta_H"]
        y_true_lab[te] = hard_labels(r[te], theta_l, theta_h)
        pred_codes = fit["pipe"].predict(X.iloc[te])
        y_pred_lab[te] = le.inverse_transform(pred_codes)
        theta_rows.append(
            {
                "fold": fold_id,
                "theta_L": theta_l,
                "theta_H": theta_h,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                **{f"train_count_{k}": v for k, v in fit["class_counts"].items()},
                "n_em_iters": len(fit["history"]),
            }
        )

    y_true = le.transform(y_true_lab)
    y_pred = le.transform(y_pred_lab)
    metrics = cls_metrics(y_true, y_pred, le)
    metrics.update(
        {
            f"{k}": v
            for k, v in nb_pb_binary_metrics(y_true_lab, y_pred_lab).items()
        }
    )
    return {
        "y_true_lab": y_true_lab,
        "y_pred_lab": y_pred_lab,
        "y_true": y_true,
        "y_pred": y_pred,
        "le": le,
        "metrics": metrics,
        "fold_thetas": theta_rows,
        "n_splits": n_splits,
    }


def write_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> None:
    labels = list(range(len(le.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in le.classes_],
        columns=[f"pred_{c}" for c in le.classes_],
    ).to_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["cf_oil", "oil_level", "job_design", "well", "well+oil", "job+oil", "ops", "all"],
        help="Feature sets to run EM on",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Force theta_H = -theta_L (single eps)",
    )
    parser.add_argument("--min-class-count", type=int, default=MIN_CLASS_COUNT)
    parser.add_argument("--n-grid", type=int, default=9, help="Quantile grid size per side")
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--C", type=float, default=1.0, help="LogReg C inside EM")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = engineer_features(compute_relative_uplift(load_merged()))
    r = merged[REL_COL].to_numpy(dtype=float)
    groups = merged["Well"]
    sets = feature_sets()

    print(
        f"n={len(merged)} jobs / {merged['Well'].nunique()} wells | "
        f"EM boundaries ({'symmetric' if args.symmetric else 'asymmetric'}) "
        f"min_class={args.min_class_count}"
    )

    metrics_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    full_em_rows: list[dict] = []
    history_rows: list[dict] = []

    id_base = merged[["ID", "Well", "Index", OBS_COL, CF_COL, REL_COL]].copy()

    for fs_name in args.feature_sets:
        if fs_name not in sets:
            raise KeyError(f"Unknown feature set: {fs_name}")
        cat_cols, num_cols = sets[fs_name]
        cols = cat_cols + num_cols
        for banned in (OBS_COL, REL_COL, LABEL_COL):
            if banned in cols:
                raise RuntimeError(f"Leakage: {banned} in {fs_name}")
        X = merged[cols].copy()

        print(f"\n=== EM OOF: {fs_name} ===")
        oof = oof_em(
            X,
            r,
            groups,
            cat_cols,
            num_cols,
            symmetric=args.symmetric,
            min_count=args.min_class_count,
            n_grid=args.n_grid,
            max_iter=args.max_iter,
            C=args.C,
        )
        m = oof["metrics"]
        row = {
            "feature_set": fs_name,
            "model": "em_logreg",
            "symmetric": bool(args.symmetric),
            "n": int(len(r)),
            "n_wells": int(groups.nunique()),
            "n_splits": int(oof["n_splits"]),
            "n_features_raw": int(X.shape[1]),
            **{f"{k}_oof": v for k, v in m.items()},
            "theta_L_mean": float(np.mean([t["theta_L"] for t in oof["fold_thetas"]])),
            "theta_H_mean": float(np.mean([t["theta_H"] for t in oof["fold_thetas"]])),
            "theta_L_std": float(np.std([t["theta_L"] for t in oof["fold_thetas"]], ddof=0)),
            "theta_H_std": float(np.std([t["theta_H"] for t in oof["fold_thetas"]], ddof=0)),
        }
        metrics_rows.append(row)
        print(
            f"  macroF1={row['macro_f1_oof']:.3f}  balAcc={row['balanced_accuracy_oof']:.3f}  "
            f"NB↔PB_bal={row['nb_pb_balanced_accuracy_oof']:.3f}  "
            f"θ≈[{row['theta_L_mean']:.3f}, {row['theta_H_mean']:.3f}]"
        )

        fold_df = pd.DataFrame(oof["fold_thetas"])
        fold_df.insert(0, "feature_set", fs_name)
        fold_df.to_csv(out_dir / f"fold_thetas_{fs_name.replace('+', '_')}.csv", index=False)

        write_confusion(
            out_dir / f"confusion_matrix_{fs_name.replace('+', '_')}_oof.csv",
            oof["y_true"],
            oof["y_pred"],
            oof["le"],
        )

        slim = id_base.copy()
        slim["feature_set"] = fs_name
        slim["y_true_em"] = oof["y_true_lab"]
        slim["yhat_oof"] = oof["y_pred_lab"]
        slim["correct"] = slim["y_true_em"] == slim["yhat_oof"]
        oof_frames.append(slim)

        # Full-data EM (reported boundaries; not used for OOF metrics)
        full = em_fit(
            X,
            r,
            cat_cols,
            num_cols,
            symmetric=args.symmetric,
            min_count=args.min_class_count,
            n_grid=args.n_grid,
            max_iter=args.max_iter,
            C=args.C,
        )
        full_em_rows.append(
            {
                "feature_set": fs_name,
                "theta_L": full["theta_L"],
                "theta_H": full["theta_H"],
                "n_candidates": full["n_candidates"],
                "n_em_iters": len(full["history"]),
                **{f"count_{k}": v for k, v in full["class_counts"].items()},
            }
        )
        for h in full["history"]:
            history_rows.append({"feature_set": fs_name, **h})

        # Attach full-data labels / memberships for this feature set's θ
        # (store once from oil_level or first set — rewrite each time; keep last / best later)
        merged[f"em_label__{fs_name}"] = full["labels"]
        mu = soft_memberships(r, full["theta_L"], full["theta_H"])
        merged[f"mu_NB__{fs_name}"] = mu[:, 0]
        merged[f"mu_ZE__{fs_name}"] = mu[:, 1]
        merged[f"mu_PB__{fs_name}"] = mu[:, 2]

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["macro_f1_oof", "balanced_accuracy_oof"], ascending=[False, False]
    )
    metrics.to_csv(out_dir / "metrics_all.csv", index=False)
    pd.DataFrame(full_em_rows).to_csv(out_dir / "full_data_em_boundaries.csv", index=False)
    if history_rows:
        # Flatten class_counts if present
        hist_flat = []
        for h in history_rows:
            flat = {k: v for k, v in h.items() if k != "class_counts"}
            cc = h.get("class_counts") or {}
            for k, v in cc.items():
                flat[f"count_{k}"] = v
            hist_flat.append(flat)
        pd.DataFrame(hist_flat).to_csv(out_dir / "em_history_full_data.csv", index=False)

    pd.concat(oof_frames, ignore_index=True).to_csv(
        out_dir / "oof_predictions_all.csv", index=False
    )

    best = metrics.iloc[0]
    summary = {
        "symmetric": bool(args.symmetric),
        "min_class_count": int(args.min_class_count),
        "best_feature_set": best["feature_set"],
        "macro_f1_oof": float(best["macro_f1_oof"]),
        "balanced_accuracy_oof": float(best["balanced_accuracy_oof"]),
        "accuracy_oof": float(best["accuracy_oof"]),
        "nb_pb_balanced_accuracy_oof": float(best["nb_pb_balanced_accuracy_oof"]),
        "theta_L_mean_oof_folds": float(best["theta_L_mean"]),
        "theta_H_mean_oof_folds": float(best["theta_H_mean"]),
        "n": int(best["n"]),
        "n_wells": int(best["n_wells"]),
    }
    # Attach full-data θ for best feature set
    full_best = next(
        r for r in full_em_rows if r["feature_set"] == best["feature_set"]
    )
    summary["full_data_theta_L"] = full_best["theta_L"]
    summary["full_data_theta_H"] = full_best["theta_H"]
    summary["full_data_counts"] = {
        "NB": full_best["count_NB"],
        "ZE": full_best["count_ZE"],
        "PB": full_best["count_PB"],
    }
    (out_dir / "best_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(out_dir / "best_summary.csv", index=False)

    merged.to_csv(out_dir / "merged_modeling_table.csv", index=False)

    # Contrast vs fixed-ε reference (same feature sets, ε=0.2) if available
    fixed_path = ROOT / "outputs" / "relative_uplift_fuzzy" / "metrics_best_per_feature_set.csv"
    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path)
        contrast = []
        for _, em_row in metrics.iterrows():
            fs = em_row["feature_set"]
            sub = fixed.loc[fixed["feature_set"] == fs]
            if sub.empty:
                continue
            contrast.append(
                {
                    "feature_set": fs,
                    "em_macro_f1_oof": em_row["macro_f1_oof"],
                    "em_balanced_accuracy_oof": em_row["balanced_accuracy_oof"],
                    "em_theta_L_mean": em_row["theta_L_mean"],
                    "em_theta_H_mean": em_row["theta_H_mean"],
                    "fixed_eps_best_model": sub.iloc[0]["model"],
                    "fixed_eps_macro_f1_oof": sub.iloc[0]["macro_f1_oof"],
                    "fixed_eps_balanced_accuracy_oof": sub.iloc[0]["balanced_accuracy_oof"],
                    "delta_macro_f1": float(
                        em_row["macro_f1_oof"] - sub.iloc[0]["macro_f1_oof"]
                    ),
                }
            )
        if contrast:
            pd.DataFrame(contrast).to_csv(
                out_dir / "contrast_vs_fixed_eps.csv", index=False
            )

    print("\n=== EM results by feature set (OOF) ===")
    cols = [
        "feature_set",
        "macro_f1_oof",
        "balanced_accuracy_oof",
        "nb_pb_balanced_accuracy_oof",
        "theta_L_mean",
        "theta_H_mean",
        "recall_NB_oof",
        "recall_ZE_oof",
        "recall_PB_oof",
    ]
    print(metrics[cols].to_string(index=False))
    print(
        f"\nBest: {summary['best_feature_set']}  "
        f"macro-F1={summary['macro_f1_oof']:.3f}  "
        f"bal-acc={summary['balanced_accuracy_oof']:.3f}"
    )
    print(
        f"  OOF-fold mean θ=[{summary['theta_L_mean_oof_folds']:.3f}, "
        f"{summary['theta_H_mean_oof_folds']:.3f}]  "
        f"full-data θ=[{summary['full_data_theta_L']:.3f}, "
        f"{summary['full_data_theta_H']:.3f}]  "
        f"counts={summary['full_data_counts']}"
    )
    print(f"Wrote artifacts under {out_dir}")


if __name__ == "__main__":
    main()
