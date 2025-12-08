# TimesFM_FX

This repository contains scripts for walk-forward forecasting experiments with **TimesFM** on EUR/NOK and a small multi-FX panel, prepared as supporting material for a master’s thesis on FX level forecasting. The project is structured for reproducible research, with scripts in `src/` and data outputs in `data/`.

## Purpose
- Benchmark **TimesFM transformer models** for EUR/NOK level forecasting.
- Compare two TimesFM generations (2.0 and 2.5) under identical walk-forward evaluation.
- Extend experiments to a **multi-FX** setting to test cross-series robustness.
- Report standard level metrics (RMSE, MAE) and directional accuracy, with Diebold–Mariano tests versus a random-walk baseline.

## Repository Structure
- `src/` — executable Python scripts for TimesFM experiments.
- `data/` — example metrics outputs produced by the runs.
- `requirements.txt` — pinned dependencies for reproducibility.
- `LICENSE` — MIT License.

## Scripts
All scripts live in `src/`. The list below maps to the thesis experiment families.

### TimesFM 2.0 (Price-Only)
- **Times2.0M.py** — TimesFM 2.0, **monthly** walk-forward evaluation (EUR/NOK).  
  Link: [`src/Times2.0M.py`](src/Times2.0M.py)

- **Times2.0.py** — TimesFM 2.0, **quarterly** walk-forward evaluation (EUR/NOK).  
  Link: [`src/Times2.0.py`](src/Times2.0.py)

### TimesFM 2.5 (Price-Only)
- **Times2.5M.py** — TimesFM 2.5, **monthly** walk-forward evaluation (EUR/NOK).  
  Link: [`src/Times2.5M.py`](src/Times2.5M.py)

- **Times2.5.py** — TimesFM 2.5, **quarterly** walk-forward evaluation (EUR/NOK).  
  Link: [`src/Times2.5.py`](src/Times2.5.py)

### TimesFM 2.5 (Multi-FX Panel)
- **Times2.5MultiM.py** — TimesFM 2.5 multi-FX batch run, **monthly** evaluation and metrics export.  
  Link: [`src/Times2.5MultiM.py`](src/Times2.5MultiM.py)

- **Times2.5Multi.py** — TimesFM 2.5 multi-FX batch run, **quarterly** evaluation and metrics export.  
  Link: [`src/Times2.5Multi.py`](src/Times2.5Multi.py)

## Data Outputs
- **Times2.5Multi_metrics_quarterly.csv** — example quarterly metrics output from the multi-FX run.  
  Link: [`data/Times2.5Multi_metrics_quarterly.csv`](data/Times2.5Multi_metrics_quarterly.csv)

## Method Summary (Thesis Context)
- **Data preprocessing:** Raw CSV series are loaded (comma/semicolon depending on source). Timestamps are parsed to a daily index (`D`) and forward-filled to ensure continuous daily inputs. Ground-truth aggregation uses business-day (`B`) reindexing to compute monthly/quarterly average levels.
- **Forecasting models:**  
  - *TimesFM 2.0* (v1 API, checkpoint `google/timesfm-2.0-500m-pytorch`).  
  - *TimesFM 2.5* (repo API, checkpoint `google/timesfm-2.5-200m-pytorch`).  
- **Walk-forward evaluation:** For each target period (month/quarter), the cut-off is the last business day of the previous period. The model consumes the most recent `N` daily observations (context) and predicts the next `H` daily values. Forecasts are aggregated to business days in the target period and averaged to obtain a predicted level.
- **Metrics and tests:** RMSE and MAE of monthly/quarterly levels, plus directional accuracy. Diebold–Mariano tests compare TimesFM to a random-walk benchmark using MSE or MAE loss.

## Quick Start
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # PowerShell (Windows)

python -m pip install -U pip
pip install -r requirements.txt
