"""Plot production time series for each acid-treatment job.

For every job ID, plots observed production vs time and overlays:

1. Free exponential fit on pre-treatment points:
       observation = a * exp(b * Index)
2. Ridge-OOF decline with fixed predicted b and scale a estimated from
   all pre-treatment points:
       observation = a_hat * exp(b_oof * Index)
       log(a_hat) = mean(log(q_i) - b_oof * Index_i)
3. Highlighted AUC regions:
   - no-shutdown [0, 180]: under the ridge-OOF curve
   - observed post [0, 180]: flat-tailed PWL on Index > 0
   - observed pre [-180, 0]: flat-tailed PWL on Index < 0

Also writes/updates exp_decline_fits.csv (preserves existing ridge columns).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_no_shutdown_oil import (  # noqa: E402
    OIL_COLS,
    T0,
    T1,
    T_PRE0,
    auc_exp_decline,
    auc_observed_pwl,
    observed_pwl_curve,
)

DEFAULT_CSV = (
    ROOT / "data" / "raw" / "production_timeseries.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "well_timeseries"
DEFAULT_FITS = DEFAULT_OUT / "exp_decline_fits.csv"

CONFIDENCE_K = 3.0
RIDGE_COLS = ("ridge_b_in_sample", "ridge_b_oof", "b_anchored")
PRESERVE_COLS = (*RIDGE_COLS, *OIL_COLS)
COLOR_NO_SHUTDOWN = "#e07a5f"
COLOR_OBSERVED = "#3d5a80"
COLOR_PRE = "#2a9d8f"
FILL_NO_SHUTDOWN = (0.88, 0.48, 0.37, 0.28)
FILL_OBSERVED = (0.24, 0.35, 0.50, 0.28)
FILL_PRE = (0.16, 0.62, 0.56, 0.28)

@dataclass
class ExpDeclineFit:
    a: float
    b: float
    n: int
    r2: float

    def predict(self, index: np.ndarray) -> np.ndarray:
        return self.a * np.exp(self.b * index)

    @property
    def confidence(self) -> float:
        return fit_confidence(self.r2, self.n)

    @property
    def label(self) -> str:
        return (
            f"Free exp fit: a={self.a:.3g}, b={self.b:.3g}/d "
            f"(R²={self.r2:.2f}, n={self.n}, conf={self.confidence:.2f})"
        )


def fit_confidence(r2: float, n: int, k: float = CONFIDENCE_K) -> float:
    if n is None or n < 2 or not np.isfinite(r2):
        return float("nan")
    r2_c = float(np.clip(r2, 0.0, 1.0))
    if n == 2:
        return 0.0
    return r2_c * ((n - 2) / (n - 2 + k))


def safe_name(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", text).strip("_")


def fit_exponential_decline(pre: pd.DataFrame) -> ExpDeclineFit | None:
    """Fit observation = a * exp(b * Index) on pre-treatment rows."""
    usable = pre.loc[pre["observation"] > 0, ["Index", "observation"]].dropna()
    if len(usable) < 2 or usable["Index"].nunique() < 2:
        return None

    x = usable["Index"].to_numpy(dtype=float)
    y = usable["observation"].to_numpy(dtype=float)
    b, log_a = np.polyfit(x, np.log(y), 1)
    a = float(np.exp(log_a))
    b = float(b)
    pred = a * np.exp(b * x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ExpDeclineFit(a=a, b=b, n=len(usable), r2=r2)


def fit_anchored_b(group: pd.DataFrame) -> tuple[float, float] | None:
    """Return (q_neg180, b_anchored) for q(t)=q180*exp(b*(t+180))."""
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
    b = float(np.dot(x[mask], y[mask]) / np.dot(x[mask], x[mask]))
    return q180, b


def scale_a_given_b(pre: pd.DataFrame, b: float) -> float | None:
    """Estimate a in q = a*exp(b*Index) from all positive pre-treatment points.

    log(a) = mean(log(q_i) - b * Index_i)
    """
    usable = pre.loc[pre["observation"] > 0, ["Index", "observation"]].dropna()
    if len(usable) == 0 or not np.isfinite(b):
        return None
    t = usable["Index"].to_numpy(dtype=float)
    q = usable["observation"].to_numpy(dtype=float)
    return float(np.exp(np.mean(np.log(q) - b * t)))


def index_to_x(
    idx_grid: np.ndarray, group: pd.DataFrame, x_col: str
) -> np.ndarray | pd.DatetimeIndex:
    if x_col == "Index":
        return idx_grid
    treat_dates = group.loc[group["Index"] == 0, "Date"]
    if len(treat_dates) == 0:
        ref = group.iloc[0]
        treat0 = ref["Date"] - pd.to_timedelta(int(ref["Index"]), unit="D")
    else:
        treat0 = treat_dates.iloc[0]
    return treat0 + pd.to_timedelta(idx_grid, unit="D")


def plot_job(
    group: pd.DataFrame,
    out_path: Path,
    x_col: str,
    ridge: dict[str, float] | None = None,
) -> ExpDeclineFit | None:
    group = group.sort_values(["Index", "Date"])
    job_id = int(group["ID"].iloc[0])
    well = str(group["Well"].iloc[0])

    pre = group.loc[group["Index"] < 0]
    fit = fit_exponential_decline(pre)

    oil_cf = float("nan")
    oil_post = auc_observed_pwl(group, side="post")
    oil_pre = auc_observed_pwl(group, side="pre")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    # Markers only — no full-span connecting line (pre/post PWL carry that role)
    ax.plot(
        group[x_col],
        group["observation"],
        marker="o",
        markersize=5,
        linestyle="None",
        color="#2c5f7c",
        label="Observed",
        zorder=4,
    )

    idx_min = float(pre["Index"].min()) if len(pre) else T_PRE0
    idx_pre = np.linspace(idx_min, 0.0, 200)

    if fit is not None:
        ax.plot(
            index_to_x(idx_pre, group, x_col),
            fit.predict(idx_pre),
            color="#b85c38",
            linewidth=1.8,
            label=fit.label,
            zorder=2,
        )

    # Ridge OOF: full curve + filled no-shutdown AUC on [0, 180]
    if ridge is not None:
        b_oof = ridge.get("ridge_b_oof")
        if b_oof is not None and np.isfinite(b_oof):
            a_hat = scale_a_given_b(pre, float(b_oof))
            if a_hat is not None:
                b = float(b_oof)
                oil_cf = auc_exp_decline(a_hat, b)
                idx_full = np.linspace(min(idx_min, T_PRE0), T1, 400)
                q_full = a_hat * np.exp(b * idx_full)
                ax.plot(
                    index_to_x(idx_full, group, x_col),
                    q_full,
                    color=COLOR_NO_SHUTDOWN,
                    linewidth=2.0,
                    linestyle="--",
                    label=(
                        f"Ridge OOF (fixed b, fit a): "
                        f"a={a_hat:.3g}, b={b:.3g}/d"
                    ),
                    zorder=2,
                )
                idx_auc = np.linspace(T0, T1, 250)
                q_auc = a_hat * np.exp(b * idx_auc)
                ax.fill_between(
                    index_to_x(idx_auc, group, x_col),
                    q_auc,
                    0.0,
                    color=FILL_NO_SHUTDOWN,
                    label=f"No-shutdown oil 0–180: {oil_cf:,.0f}",
                    zorder=1,
                )

    # Observed pre-shutdown PWL with flat tails + filled AUC
    knots_pre = observed_pwl_curve(group, side="pre")
    if knots_pre is not None:
        t_pwl, q_pwl = knots_pre
        ax.plot(
            index_to_x(t_pwl, group, x_col),
            q_pwl,
            color=COLOR_PRE,
            linewidth=2.0,
            linestyle="-",
            label="Pre PWL (flat tails)",
            zorder=3,
        )
        t_fill = np.linspace(T_PRE0, T0, 300)
        q_fill = np.interp(t_fill, t_pwl, q_pwl)
        ax.fill_between(
            index_to_x(t_fill, group, x_col),
            q_fill,
            0.0,
            color=FILL_PRE,
            label=f"Observed oil −180–0: {oil_pre:,.0f}",
            zorder=1,
        )

    # Observed post-treatment PWL with flat tails + filled AUC
    knots_post = observed_pwl_curve(group, side="post")
    if knots_post is not None:
        t_pwl, q_pwl = knots_post
        ax.plot(
            index_to_x(t_pwl, group, x_col),
            q_pwl,
            color=COLOR_OBSERVED,
            linewidth=2.0,
            linestyle="-",
            label="Post PWL (flat tails)",
            zorder=3,
        )
        t_fill = np.linspace(T0, T1, 300)
        q_fill = np.interp(t_fill, t_pwl, q_pwl)
        ax.fill_between(
            index_to_x(t_fill, group, x_col),
            q_fill,
            0.0,
            color=FILL_OBSERVED,
            label=f"Observed oil 0–180: {oil_post:,.0f}",
            zorder=1,
        )

    treat = group.loc[group["Index"] == 0]
    if len(treat):
        ax.axvline(
            treat[x_col].iloc[0],
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            label="Treatment",
            zorder=1,
        )
    ax.axvline(
        index_to_x(np.array([T_PRE0]), group, x_col)[0],
        color="#aaaaaa",
        linestyle=":",
        linewidth=1.0,
        label="Index = −180",
        zorder=0,
    )
    ax.axvline(
        index_to_x(np.array([T1]), group, x_col)[0],
        color="#aaaaaa",
        linestyle=":",
        linewidth=1.0,
        label="Index = 180",
        zorder=0,
    )

    lines = []
    if np.isfinite(oil_pre):
        lines.append(f"pre observed AUC  = {oil_pre:,.1f}")
    if np.isfinite(oil_cf):
        lines.append(f"no-shutdown AUC   = {oil_cf:,.1f}")
    if np.isfinite(oil_post):
        lines.append(f"post observed AUC = {oil_post:,.1f}")
    if lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "alpha": 0.9,
            },
            zorder=5,
        )

    ax.set_xlabel(
        "Index (days relative to treatment)" if x_col == "Index" else "Date"
    )
    ax.set_ylabel("observation")
    ax.set_title(f"{well}  (job ID {job_id})")
    ax.legend(loc="best", frameon=False, fontsize=7.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if x_col == "Date":
        fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--x",
        choices=("Index", "Date"),
        default="Index",
        help="Horizontal axis: relative day Index (default) or calendar Date",
    )
    parser.add_argument(
        "--fits-csv",
        type=Path,
        default=None,
        help="Fit/ridge table path (default: <out-dir>/exp_decline_fits.csv)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only compute fits CSV (skip writing PNGs)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["Date"])
    required = {"ID", "Well", "Date", "Index", "observation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits_path = args.fits_csv or (args.out_dir / "exp_decline_fits.csv")

    # Load prior ridge / oil columns so regenerating free fits does not wipe them
    prior_extra = pd.DataFrame()
    if fits_path.exists():
        prior = pd.read_csv(fits_path)
        keep = [c for c in PRESERVE_COLS if c in prior.columns]
        if keep:
            prior_extra = prior[["ID", *keep]].copy()

    ridge_by_id: dict[int, dict[str, float]] = {}
    if not prior_extra.empty:
        for _, row in prior_extra.iterrows():
            ridge_by_id[int(row["ID"])] = {
                c: float(row[c]) if pd.notna(row[c]) else float("nan")
                for c in RIDGE_COLS
                if c in prior_extra.columns
            }

    rows: list[dict] = []
    n = 0
    n_fit = 0
    n_ridge = 0
    for job_id, group in df.groupby("ID", sort=True):
        well = str(group["Well"].iloc[0])
        job_id_i = int(job_id)
        pre = group.loc[group["Index"] < 0]
        fit = fit_exponential_decline(pre)
        anchored = fit_anchored_b(group)
        ridge = ridge_by_id.get(job_id_i)

        if not args.no_plots:
            fname = f"job_{job_id_i:03d}_{safe_name(well)}.png"
            plot_job(group, args.out_dir / fname, args.x, ridge=ridge)
            if ridge and any(
                np.isfinite(ridge.get(c, np.nan))
                for c in ("ridge_b_oof", "ridge_b_in_sample")
            ):
                n_ridge += 1

        n += 1
        row: dict = {"ID": job_id_i, "Well": well}
        if fit is not None:
            n_fit += 1
            row.update(
                {
                    "a": fit.a,
                    "b": fit.b,
                    "n": fit.n,
                    "r2": fit.r2,
                    "confidence": fit.confidence,
                }
            )
        else:
            row.update(
                {
                    "a": np.nan,
                    "b": np.nan,
                    "n": 0,
                    "r2": np.nan,
                    "confidence": np.nan,
                }
            )
        if anchored is not None:
            row["b_anchored"] = anchored[1]
        else:
            row["b_anchored"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)

    # Re-attach ridge / oil columns from prior file (do not overwrite with NaN)
    if not prior_extra.empty:
        ridge_only = [
            c for c in ("ridge_b_in_sample", "ridge_b_oof") if c in prior_extra.columns
        ]
        if ridge_only:
            out = out.drop(columns=ridge_only, errors="ignore")
            out = out.merge(prior_extra[["ID", *ridge_only]], on="ID", how="left")
        # Prefer freshly computed b_anchored; fill gaps from prior
        if "b_anchored" in prior_extra.columns:
            out["b_anchored"] = out["b_anchored"].fillna(
                out["ID"].map(prior_extra.set_index("ID")["b_anchored"])
            )
        oil_keep = [c for c in OIL_COLS if c in prior_extra.columns]
        if oil_keep:
            out = out.drop(columns=oil_keep, errors="ignore")
            out = out.merge(prior_extra[["ID", *oil_keep]], on="ID", how="left")

    # Stable column order
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
    out = out[cols].sort_values("ID")
    out.to_csv(fits_path, index=False)

    if args.no_plots:
        print(f"Skipped plots; computed fits for {n} jobs ({n_fit} with exp fits)")
    else:
        print(
            f"Wrote {n} figures ({n_fit} with exp fits, "
            f"{n_ridge} with ridge overlays) to {args.out_dir}"
        )
    print(f"Wrote fit parameters to {fits_path}")


if __name__ == "__main__":
    main()
