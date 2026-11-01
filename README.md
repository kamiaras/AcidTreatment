# AcidTreatment

Python tools for exploring relationships between well characteristics, treatment parameters, and production curves. The code includes decline fitting, regression grouped by well, and fuzzy classes for relative uplift.

This code-only edition contains no field data, fitted models, original results, or field figures. The example uses invented values. It demonstrates selected functions, not treatment effectiveness or a complete field study.

## Run the synthetic example

Tested with Python 3.12.14 on Windows. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python examples/synthetic_demo.py
```

The example checks grouped regression, treatment ordering, fuzzy-label boundaries, an exponential fit, and curve integrals. It prints a report and removes its temporary files when finished. To keep synthetic results in a new directory:

```bash
python examples/synthetic_demo.py --output-dir artifacts/synthetic-run
```

An existing output directory is refused. The pinned Windows dependency list is in [requirements-lock-windows-py312.txt](requirements-lock-windows-py312.txt); other platforms have not been tested.

## Use your own inputs

The analysis scripts require your own private input files. Start with the [input schema and execution order](docs/INPUT_SCHEMA.md). Input preparation from a particular company's raw export is not included.

Keep inputs and generated results local. The `data/`, `outputs/`, `models/`, `reports/`, and `artifacts/` directories are ignored by Git. Some scripts overwrite their output files; the optional imputation script also overwrites its input job workbook. Make a separate working copy before using it.

## Contents and limits

- [scripts/](scripts/) contains reusable functions and file-based analysis scripts.
- [examples/synthetic_demo.py](examples/synthetic_demo.py) creates and checks invented examples.
- [Verification and method limits](docs/REPRODUCIBILITY.md) explains what was tested.
- [Data provenance](docs/DATA_PROVENANCE.md) explains the code-only release.

The baseline is a model estimate, not an observed untreated control. Predictive scores and uplift labels do not establish a causal treatment effect. Review assumptions, units, joins, and validation design before interpreting results on your own data.

Maintained by Kamiar Asgari. This edition retains the existing analysis code and adds a synthetic example and documentation. No software license has been selected for this repository.
