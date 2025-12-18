# -*- coding: utf-8 -*-
"""
TimesFM 2.0 (500M, PyTorch) – EUR/NOK walk-forward (monthly) using shared evaluation.
Source: GitHub all-variables daily panel (comma-separated).

Shared evaluation (eval_common.py) ensures:
- Same cut definition
- Same monthly target (mean over business days)
- Same driftless RW benchmark for DM: rw_pred = S_b.loc[cut]
- Same metrics + DM + plotting
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

# Prefer local TimesFM v1 API (timesfm/v1/src) over workspace's newer API.
# This avoids import conflicts with the top-level `timesfm` package.
_ROOT = Path(__file__).resolve().parent.parent
_V1_SRC = _ROOT / "timesfm" / "v1" / "src"
if _V1_SRC.exists():
    sys.path.insert(0, str(_V1_SRC))

import timesfm  # v1 API (TimesFm, TimesFmHparams, TimesFmCheckpoint)

from eval_common import (
    EvalConfig,
    load_series,
    walk_forward_monthly,
    evaluate_monthly,
    dm_against_rw,
    plot_monthly,
)

# -----------------------------
# Config
# -----------------------------
CFG = EvalConfig(
    url=(
        "https://raw.githubusercontent.com/bredeespelid/"
        "Data_MasterOppgave/refs/heads/main/Variables/All_Variables/variables_daily.csv"
    ),
    series="EUR_NOK",
    m_freq="M",
    min_hist_days=40,
    max_context=2048,
    max_horizon=64,
    retries=3,
    timeout=60,
    verbose=True,
)

FIG_PNG = "data/Times2.0M.png"
FIG_PDF = "data/Times2.0M.pdf"


# -----------------------------
# Model (TimesFM 2.0 – 500M, PyTorch) v1 API
# -----------------------------
def build_timesfm20(horizon_len: int):
    required = all([
        hasattr(timesfm, "TimesFm"),
        hasattr(timesfm, "TimesFmHparams"),
        hasattr(timesfm, "TimesFmCheckpoint"),
    ])
    if not required:
        raise RuntimeError(
            "TimesFM v1 API not found. Install 'timesfm==1.3.0' for 2.0-500M:\n"
            "  python -m pip install timesfm==1.3.0"
        )

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=32,
            horizon_len=horizon_len,   # must cover longest month (<=31); 64 safe
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    return tfm


def make_forecast_daily_fn(tfm):
    """
    Adapter to the shared eval interface:
      forecast_daily_fn(x_1d, H) -> np.ndarray of length H (daily point forecast)
    """
    def _forecast_daily(x_1d: np.ndarray, H: int) -> np.ndarray:
        point_forecast, _ = tfm.forecast([x_1d], freq=[0])
        pf = np.asarray(point_forecast[0], dtype=float)
        return pf[:H]
    return _forecast_daily


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Data (shared)
    S_b, S_d = load_series(CFG)
    if CFG.verbose:
        print(f"Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")
        print(f"Data (D): {S_d.index.min().date()} → {S_d.index.max().date()} | n={len(S_d)}")

    # 2) Model
    tfm = build_timesfm20(horizon_len=CFG.max_horizon)
    forecast_daily_fn = make_forecast_daily_fn(tfm)

    # 3) Walk-forward (shared) + metrics (shared)
    df_eval = walk_forward_monthly(S_b, S_d, CFG, forecast_daily_fn)
    eval_df = evaluate_monthly(df_eval)

    # 4) DM vs driftless RW (shared, consistent with RW baseline)
    dm_against_rw(eval_df, loss="mse", h=1)

    # 5) Plot (shared)
    plot_monthly(
        eval_df,
        title="TimesFM 2.0 Forecast vs Actual (Monthly Mean, EUR/NOK)",
        png_path=FIG_PNG,
        pdf_path=FIG_PDF,
    )


if __name__ == "__main__":
    main()
