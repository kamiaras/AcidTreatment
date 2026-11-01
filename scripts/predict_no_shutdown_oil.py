"""Oil AUCs: counterfactual (ridge OOF) and observed PWL (pre / post).

Counterfactual (no shut-in), using ridge_b_oof:
    q(t) = a_hat * exp(b * t)
    log(a_hat) = mean(log(q_i) - b * Index_i)   # positive pre-treatment points
    oil_0_180_no_shutdown = ∫_0^180 q(t) dt

Counterfactual (no shut-in), using free-fit a,b from exp_decline_fits
(CSV only; not plotted):
    oil_0_180_no_shutdown_free = ∫_0^180 a_free * exp(b_free * t) dt

Observed post-treatment oil (Index > 0):
    Piecewise-linear through observations with Index > 0.
    Left tail  [0, first_t]:  flat at first post observation.
    Right tail [last_t, 180]: flat at last post observation.
    oil_0_180_observed = ∫_0^180 of that PWL curve
    → written to observed_oil_0_180.csv (not no_shutdown_oil.csv)

Observed pre-shutdown oil (Index < 0):
    Piecewise-linear through observations with Index < 0.
    Left tail  [-180, first_t]: flat at first pre observation.
    Right tail [last_t, 0]:     flat at last pre observation.
    oil_neg180_0_observed = ∫_{-180}^0 of that PWL curve.

Writes oil columns into exp_decline_fits.csv (merged; no separate
no_shutdown_oil.csv) and observed_oil_0_180.csv for post observed AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TS = (
    ROOT / "data" / "raw" / "production_timeseries.csv"
)
DEFAULT_FITS = ROOT / "outputs" / "well_timeseries" / "exp_decline_fits.csv"
DEFAULT_OBS_OUT = (
    ROOT / "outputs" / "well_timeseries" / "observed_oil_0_180.csv"
)

OIL_COLS = (
    "a_hat",
    "oil_0_180_no_shutdown",
    "oil_0_180_no_shutdown_free",
    "oil_neg180_0_observed",
)
T_PRE0 = -180.0
T0 = 0.0
T1 = 180.0


def scale_a_given_b(pre: pd.DataFrame, b: float) -> float | None:
    """Estimate a in q = a*exp(b*Index) from all positive pre-treatment points."""
    usable = pre.loc[pre["observation"] > 0, ["Index", "observation"]].dropna()
    if len(usable) == 0 or not np.isfinite(b):
        return None
    t = usable["Index"].to_numpy(dtype=float)
    q = usable["observation"].to_numpy(dtype=float)
    return float(np.exp(np.mean(np.log(q) - b * t)))


def auc_exp_decline(a: float, b: float, t0: float = T0, t1: float = T1) -> float:
    """Analytic ∫_{t0}^{t1} a * exp(b t) dt."""
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    if abs(b) < 1e-15:
        return float(a * (t1 - t0))
    return float((a / b) * (np.exp(b * t1) - np.exp(b * t0)))


def observed_pwl_curve(
    group: pd.DataFrame,
    *,
    side: str = "post",
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Knots (t, q) for flat-tailed PWL over [t0, t1].

    side='post': use Index > 0, default window [0, 180]
    side='pre':  use Index < 0, default window [-180, 0]
    """
    if side == "post":
        mask = group["Index"] > 0
        if t0 is None:
            t0 = T0
        if t1 is None:
            t1 = T1
    elif side == "pre":
        mask = group["Index"] < 0
        if t0 is None:
            t0 = T_PRE0
        if t1 is None:
            t1 = T0
    else:
        raise ValueError(f"side must be 'pre' or 'post', got {side!r}")

    pts = (
        group.loc[mask, ["Index", "observation"]]
        .dropna()
        .sort_values("Index")
    )
    if len(pts) == 0:
        return None

    t = pts["Index"].to_numpy(dtype=float)
    q = pts["observation"].to_numpy(dtype=float)

    keep = (t >= t0) & (t <= t1)
    t_in = t[keep]
    q_in = q[keep]
    if len(t_in) == 0:
        q_left = float(q[0])
        q_right = float(q[-1])
        return (
            np.array([t0, t1], dtype=float),
            np.array([q_left, q_right], dtype=float),
        )

    ts: list[float] = [float(t0)]
    qs: list[float] = [float(q_in[0])]
    for ti, qi in zip(t_in, q_in):
        ti_f, qi_f = float(ti), float(qi)
        if abs(ti_f - ts[-1]) < 1e-12:
            qs[-1] = qi_f
        else:
            ts.append(ti_f)
            qs.append(qi_f)
    if ts[-1] < t1 - 1e-12:
        ts.append(float(t1))
        qs.append(float(q_in[-1]))
    elif abs(ts[-1] - t1) > 1e-12:
        ts[-1] = float(t1)

    return np.asarray(ts, dtype=float), np.asarray(qs, dtype=float)


def auc_observed_pwl(
    group: pd.DataFrame,
    *,
    side: str = "post",
    t0: float | None = None,
    t1: float | None = None,
) -> float:
    """∫ of flat-tailed piecewise-linear rate over the chosen window."""
    knots = observed_pwl_curve(group, side=side, t0=t0, t1=t1)
    if knots is None:
        return float("nan")
    t, q = knots
    return float(np.trapezoid(q, t))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeseries", type=Path, default=DEFAULT_TS)
    parser.add_argument("--fits-csv", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--obs-out-csv", type=Path, default=DEFAULT_OBS_OUT)
    args = parser.parse_args()

    ts = pd.read_csv(args.timeseries)
    fits = pd.read_csv(args.fits_csv)
    if "ridge_b_oof" not in fits.columns:
        raise SystemExit(f"Missing ridge_b_oof in {args.fits_csv}")

    oil_rows: list[dict] = []
    obs_rows: list[dict] = []
    for _, fit_row in fits.iterrows():
        job_id = int(fit_row["ID"])
        well = str(fit_row["Well"])
        b = float(fit_row["ridge_b_oof"])
        a_free = float(fit_row["a"]) if pd.notna(fit_row["a"]) else float("nan")
        b_free = float(fit_row["b"]) if pd.notna(fit_row["b"]) else float("nan")
        group = ts.loc[ts["ID"] == job_id]
        pre = group.loc[group["Index"] < 0]
        a_hat = scale_a_given_b(pre, b)
        oil_cf = (
            auc_exp_decline(a_hat, b) if a_hat is not None else float("nan")
        )
        oil_cf_free = auc_exp_decline(a_free, b_free)
        oil_post = auc_observed_pwl(group, side="post")
        oil_pre = auc_observed_pwl(group, side="pre")
        oil_rows.append(
            {
                "ID": job_id,
                "a_hat": a_hat if a_hat is not None else float("nan"),
                "oil_0_180_no_shutdown": oil_cf,
                "oil_0_180_no_shutdown_free": oil_cf_free,
                "oil_neg180_0_observed": oil_pre,
            }
        )
        obs_rows.append(
            {
                "ID": job_id,
                "Well": well,
                "oil_0_180_observed": oil_post,
            }
        )

    oil = pd.DataFrame(oil_rows)
    obs = pd.DataFrame(obs_rows).sort_values("ID").reset_index(drop=True)

    # Merge oil columns into fits; drop any prior copies of those columns
    fits = fits.drop(columns=[c for c in OIL_COLS if c in fits.columns], errors="ignore")
    out = fits.merge(oil, on="ID", how="left").sort_values("ID").reset_index(drop=True)
    preferred = [
        "ID",
        "Well",
        "a",
        "b",
        "n",
        "r2",
        "confidence",
        "b_anchored",
        "ridge_b_in_sample",
        "ridge_b_oof",
        *OIL_COLS,
    ]
    cols = [c for c in preferred if c in out.columns] + [
        c for c in out.columns if c not in preferred
    ]
    out = out[cols]

    args.fits_csv.parent.mkdir(parents=True, exist_ok=True)
    args.obs_out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.fits_csv, index=False)
    obs.to_csv(args.obs_out_csv, index=False)

    # Remove legacy separate no-shutdown file if present
    legacy = args.fits_csv.parent / "no_shutdown_oil.csv"
    if legacy.exists():
        legacy.unlink()
        print(f"Removed {legacy}")

    print(f"Wrote merged fits+oil to {args.fits_csv}")
    print(out[list(OIL_COLS)].describe().to_string())
    print(
        f"Wrote {args.obs_out_csv} "
        f"(oil_0_180_observed={int(obs['oil_0_180_observed'].notna().sum())}"
        f"/{len(obs)})"
    )


if __name__ == "__main__":
    main()
