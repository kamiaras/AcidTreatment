# Use your own inputs

Run commands from the repository root in an environment with `requirements.txt` installed. No input data is distributed. Create the following local files from data you are allowed to use.

## Input files

| Path | Required columns and meaning |
| --- | --- |
| `data/raw/production_timeseries.csv` | `ID`: numeric treatment ID; `Well`: well identifier; `Date`: parseable date; `Index`: numeric day relative to treatment; `observation`: numeric production rate. |
| `data/processed/well_characteristics.xlsx` | `Index`: numeric treatment index; `Well`; categorical columns `Crude`, `Field`, `Reservoir`, `Payzone`, `WellType`, `Artificial Lift`. |
| `data/processed/job_parameters.xlsx` | `Index`; `Well`; `Vendor`, `Type of Intervention`, `Acid Concentration %`, `Total Acid Volume`, `Total Stim Pay ft`, `Dosage (Gal/ft)`. |

The two workbooks must have the same ordered `Index` column. Their well identifiers must refer to the same treatments. The job imputation and well-to-job regression code treats vendor and intervention as numeric codes; well categories are converted to strings for one-hot encoding.

Time-series `ID` and workbook `Index` are separate identifiers. Joins match `Well`, sort time-series treatments by `ID` and workbook treatments by `Index`, then pair by position within each well. You must make those orders agree. Do not rely on the code to discover the correct treatment correspondence.

Use days for time. Use one consistent production-rate unit, so integrals over days have the corresponding production-volume unit. The imputation code assumes acid volume is in barrels and stimulated pay is in feet when it applies `Dosage (Gal/ft) = 42 * Total Acid Volume / Total Stim Pay ft`. Confirm those units before using that step.

Pre-treatment decline fitting uses positive observations at negative relative days. Anchored fits also need a positive observation at `Index = -180`. Grouped validation requires multiple distinct wells; classification also needs adequate examples in each class.

## Execution order

These are the file-based entry points and their dependencies, not a claim that the full workflow was tested on your data.

1. Prepare and validate both workbooks and the time-series CSV yourself. No general raw-export converter is included.
2. Create free decline fits:

   ```bash
   python scripts/plot_well_timeseries.py --no-plots
   ```

3. Fit well-to-decline models and add the held-out ridge estimate to the fits table:

   ```bash
   python scripts/regress_b_from_well.py
   ```

4. Compute observed and estimated baseline integrals:

   ```bash
   python scripts/predict_no_shutdown_oil.py
   ```

5. Run one analysis, for example:

   ```bash
   python scripts/regress_observed_oil.py --target relative
   python scripts/classify_relative_uplift_fuzzy.py
   python scripts/classify_relative_uplift_fuzzy_em.py
   ```

The final three commands are separate alternatives, not mandatory stages. Regression also accepts `--target absolute` and `--target volume_norm`. The classifiers and regression can take longer than the small synthetic example.

Generated tables go under `outputs/`. The integral step updates `outputs/well_timeseries/exp_decline_fits.csv`, writes `observed_oil_0_180.csv`, and removes an older `no_shutdown_oil.csv` in that output directory if present. To draw time-series plots after these stages, rerun `plot_well_timeseries.py` without `--no-plots`.

## Optional workbook analysis

`python scripts/linear_regression_well_to_job.py` fits well characteristics to job parameters and writes predictions and metrics under `outputs/well_to_job_lr/`.

`python scripts/impute_job_parameters.py` fills missing or invalid job values and **overwrites** `data/processed/job_parameters.xlsx`. It first reads `outputs/well_to_job_lr/job_parameters_before_imputation.xlsx` if that file already exists; otherwise it reads the job workbook. It does not create that backup for you. Back up the workbook yourself, inspect the assumptions in the script, and check all generated imputation logs before accepting the result.

For a quick run that needs no input files, use `python examples/synthetic_demo.py` instead.
