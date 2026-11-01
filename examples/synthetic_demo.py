"""Test the existing well-grouped regression and uplift labels using invented rows."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from regress_observed_oil import build_pipeline, eval_combo, zip_join_on_well
from classify_relative_uplift_fuzzy import assign_hard_labels, triangular_memberships
from plot_well_timeseries import fit_exponential_decline
from predict_no_shutdown_oil import auc_exp_decline, auc_observed_pwl


def run(output: Path) -> dict:
    rng = np.random.default_rng(7)
    n = 30
    groups = pd.Series([f"synthetic-well-{i // 3:02d}" for i in range(n)])
    dose = rng.uniform(1, 4, n)
    features = pd.DataFrame({"synthetic_dose": dose,
                             "synthetic_category": np.where(np.arange(n) % 2, "A", "B")})
    target = 0.5 * dose + rng.normal(0, 0.05, n)
    pipeline = build_pipeline(["synthetic_category"], ["synthetic_dose"], "scaled", Ridge(alpha=1.0), y_transform="none")
    metrics, fitted, held_out = eval_combo("ridge", "synthetic", pipeline, features, target, groups, y_transform="none")
    assert metrics["n_splits"] == 5 and metrics["n_wells"] == 10
    assert fitted.shape == held_out.shape == (n,) and np.isfinite(held_out).all()

    # Verify the existing retreatment join: match by well and within-well order, not equal IDs.
    left = pd.DataFrame({"Well": ["synthetic-A", "synthetic-A", "synthetic-B"], "ID": [20, 10, 30]})
    right = pd.DataFrame({"Well": ["synthetic-B", "synthetic-A", "synthetic-A"], "Index": [9, 8, 7]})
    joined = zip_join_on_well(left, right, left_order="ID", right_order="Index")
    assert joined[["ID", "Index"]].values.tolist() == [[10, 7], [20, 8], [30, 9]]
    uplift = np.array([-0.8, -0.2, 0.0, 0.2, 0.8])
    labels = assign_hard_labels(uplift, 0.2)
    assert labels.tolist() == ["NB", "ZE", "ZE", "ZE", "PB"]
    memberships = triangular_memberships(uplift, 0.2)
    assert np.isfinite(memberships.to_numpy()).all()
    assert np.allclose(memberships.sum(axis=1), 1.0)
    times = np.array([-180.0, -120.0, -60.0, -1.0])
    decline = pd.DataFrame({"Index": times, "observation": 10.0 * np.exp(-0.002 * times)})
    fitted_decline = fit_exponential_decline(decline)
    assert fitted_decline is not None
    assert np.allclose([fitted_decline.a, fitted_decline.b], [10.0, -0.002])
    assert np.isclose(auc_exp_decline(10.0, 0.0), 1800.0)
    flat_curve = pd.DataFrame({"Index": [10.0, 90.0, 170.0], "observation": [10.0, 10.0, 10.0]})
    assert np.isclose(auc_observed_pwl(flat_curve), 1800.0)
    frame = features.assign(synthetic_well=groups, synthetic_target=target, synthetic_oof_prediction=held_out)
    frame.to_csv(output / "synthetic-grouped-regression.csv", index=False)
    report = {"synthetic": True, "seed": 7, "rows": n, "well_groups": 10,
              "synthetic_metrics": metrics,
              "checks": "five-fold grouped regression; retreatment join ordering; boundary labels; normalized memberships; known exponential fit and constant-curve integrals",
              "scope": "No proprietary files are read. Exercises reusable functions, not the complete private-data pipeline or treatment efficacy."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory for synthetic predictions and report.")
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = run(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="acid-synthetic-") as folder:
            report = run(Path(folder))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
