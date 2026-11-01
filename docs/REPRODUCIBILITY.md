# Verification and method limits

## Verified synthetic example

The example was tested with Python 3.12.14 on Windows using the versions in `requirements.txt`. It does not load field data. It checks:

- Five-fold regression with invented wells kept in separate validation groups.
- Matching repeated treatments by well and within-well order.
- Fuzzy-label boundaries and normalized memberships.
- An exponential fit with known parameters.
- Integrals of constant-rate curves.

The example's saved-output mode writes only to a new directory and refuses an existing one. These checks cover selected functions. They do not validate every model, full file-based workflow, or scientific conclusion.

## Using the analysis scripts

The scripts need user-supplied files described in [INPUT_SCHEMA.md](INPUT_SCHEMA.md). The full field-data workflow has not been rerun for this code-only edition. Many scripts replace previous outputs at their default paths. Use a separate working directory for each analysis and back up inputs before optional imputation.

The default decline and integration windows are analysis choices. Check whether they suit your study. In particular, anchored fits require a positive observation at relative day -180, and observed-curve integration uses constant values outside available knots within its requested window.

Repeated treatments are joined by well and sorted within-well order. The join can silently pair the wrong treatments if the two source orders do not correspond, and inner joins drop unmatched rows. Check the merged table before fitting models.

Grouped validation keeps wells apart in the outer split. Some engineered features are calculated before that split, and some model-selection routines use their own internal validation. This workflow is not a fully nested, leakage-audited evaluation. It does not establish the effect of a treatment or predict performance on a new field.

Missing values, too few wells, absent classes, and incompatible units can cause errors or invalid conclusions. The scripts are research tools, not a validated operational decision system.
