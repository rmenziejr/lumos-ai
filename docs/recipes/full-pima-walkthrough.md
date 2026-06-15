# Full Pima Walkthrough

The full example notebook uses the Pima Indians Diabetes dataset to walk through a
complete local `lumosai` workflow:

- raw and preprocessed data profiling;
- representative train and holdout samples;
- holdout performance with probability scores and lift;
- calibration metrics before and after probability calibration;
- bias slices by derived age group;
- benchmark and previous-window drift;
- training and monitoring bundles.

Open the notebook in GitHub:

[Pima diabetes monitoring walkthrough](https://github.com/rmenziejr/lumos-ai/blob/main/examples/notebooks/pima_diabetes_walkthrough.ipynb)

The notebook is committed with executed output examples, including embedded HTML
profile output where notebook viewers support rich HTML.

The source CSV is checked into the repo at
[`examples/data/diabetes.csv`](https://github.com/rmenziejr/lumos-ai/blob/main/examples/data/diabetes.csv)
so the notebook can run without Kaggle credentials.

Run it locally from the repository root:

```bash
uv sync --extra dev
uv run --with notebook jupyter notebook examples/notebooks/pima_diabetes_walkthrough.ipynb
```

The walkthrough uses `age_group` for the bias example. The Pima dataset does not
include gender, and the cohort is women, so the notebook avoids a fabricated
gender bias report.
