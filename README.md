# TIMES (Master thesis experiments)

This repository contains scripts used for walk-forward forecasting experiments with TimesFM for a master's thesis. The scripts are grouped into `src/` and data into `data/` to make the project ready for committing.

Contents
- `src/Times2.0.py`, `src/Times2.0M.py` – TimesFM 2.0 (v1 API) examples for quarterly and monthly evaluation.
- `src/Times2.5.py`, `src/Times2.5M.py` – TimesFM 2.5 examples for quarterly and monthly evaluation (repo API).
- `src/Times2.5Multi.py` – Multi-FX quarterly batch run and metrics export.
- `data/Times2.5Multi_metrics_quarterly.csv` – Example metrics output (quarterly) produced by the multi-FX run.

Method summary (for thesis)
- Data preprocessing: raw GitHub CSV files are read (semicolon or comma depending on source). Time stamps are parsed to a daily index (`D`) and forward-filled to ensure continuous daily input to the models. Business-day (`B`) reindexing is used for ground-truth aggregation (quarterly/monthly averages over business days).
- Forecasting method: TimesFM transformer-based time series model. Two families are included in experiments:
  - TimesFM 2.0 (v1 API; checkpoint: `google/timesfm-2.0-500m-pytorch`).
  - TimesFM 2.5 (repo API; checkpoint: `google/timesfm-2.5-200m-pytorch`).
- Walk-forward evaluation: for each period (quarter or month) the cut is defined as last B-day in the previous period; model receives the most recent N days (context) and forecasts the next H daily values; forecast daily series are aggregated to B-days in the target period and averaged to produce the predicted level.
- Metrics and statistical tests: reported metrics include RMSE and MAE of quarterly/monthly levels, and directional accuracy. Diebold–Mariano tests compare TimesFM performance to a random-walk benchmark (previous period level) using MSE or MAE loss.

How to run (quick)
1. Create a virtual environment and install core packages:

   python -m venv .venv
   .venv\Scripts\Activate.ps1  # PowerShell on Windows
   python -m pip install -U pip
   python -m pip install -r requirements.txt

2. Install TimesFM (2.5 experiments require the repo install):
   - For TimesFM 2.5: clone https://github.com/google-research/timesfm && pip install -e .
   - For TimesFM 2.0 v1 API: pip install timesfm==1.3.0

3. Run a script, for example quarterly TimesFM 2.5:

   python src/Times2.5.py

Notes and assumptions
- Comments and original logic in the scripts were preserved. I did not alter algorithmic comments or change computation steps.
- The repository does not include large model weights or the TimesFM repo. The scripts expect TimesFM to be installed separately.

Next steps you might want
- Add unit tests or a small smoke test that runs data loading and a fast deterministic forecast function.
- Add a simple runner (Makefile or CLI) to orchestrate experiments.
