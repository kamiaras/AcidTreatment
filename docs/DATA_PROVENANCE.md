# Data provenance

This release contains source code, dependency lists, documentation, and a synthetic example. It contains no original field records, processed tables, fitted parameters, model files, evaluation tables, or field figures.

The values and well names in `examples/synthetic_demo.py` are invented. The example generates them from a fixed random seed or simple formulas. They do not summarize or reproduce a field dataset.

The file and column names in the input guide describe the interface expected by the code. They are not data records. Analysis windows, numeric tolerances, model search grids, and the default classification threshold are code settings, not reported measurements or fitted field results.

If you use your own data, keep both inputs and derived outputs private unless you have permission to share them. Outputs can include input rows, well identifiers, predictions, fitted coefficients, and plots. The repository's ignore rules cover the default data and output directories; they do not protect files saved elsewhere or files added by force.

The original working material is preserved separately and is not part of this code-only release. No data license or new software license is granted by this documentation.
