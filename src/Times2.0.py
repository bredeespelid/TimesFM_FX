# -*- coding: utf-8 -*-
"""
TimesFM 2.0 (500M, PyTorch) – EUR/NOK walk-forward (quarterly) using eval_common_q.py
Source: Norges Bank CSV (semicolon separated, decimal comma)
No intervals (point forecast only).
"""

from __future__ import annotations

import numpy as np
import timesfm  # v1 API (timesfm==1.3.0)

from eval_common_quarterly import (
    EvalConfigQ,
    load_series_q,
    walk_forward_q,
    evaluate_q,
    dm_against_rw_q,
    plot_q,
)


# -----------------------------
# Config (eval_common_q)
# -----------------------------
CFG = EvalConfigQ(
    url="https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/refs/heads/main/EURNOK/EUR_NOK_NorgesBank.csv",
    series="EUR_NOK",
    q_freq="Q-DEC",
    min_hist_days=40,
    max_context=2048,
    max_horizon=512,
    retries=3,
    timeout=60,
    verbose=True,
    # Norges Bank CSV dialect:
    date_col="TIME_PERIOD",
    value_col="OBS_VALUE",     # long format: TIME_PERIOD + OBS_VALUE
    csv_sep=";",
    csv_decimal=",",
    csv_encoding="utf-8-sig",
)

FIG_PNG = "EUR_NOK_TimesFM20_Q.png"
FIG_PDF = "EUR_NOK_TimesFM20_Q.pdf"


# -----------------------------
# TimesFM 2.0 model builder (v1)
# -----------------------------
def build_timesfm20(horizon_len: int):
    required = all([
        hasattr(timesfm, "TimesFm"),
        hasattr(timesfm, "TimesFmHparams"),
        hasattr(timesfm, "TimesFmCheckpoint"),
    ])
    if not required:
        raise RuntimeError(
            "TimesFM v1 API not found. Install 'timesfm==1.3.0' for TimesFM 2.0:\n"
            "  python -m pip install timesfm==1.3.0"
        )

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=32,
            horizon_len=horizon_len,   # must cover longest quarter in calendar days (~92)
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
    Adapter expected by eval_common_q:
      forecast_daily_fn(x_1d, H) -> np.ndarray length H
    """
    def _forecast_daily(x_1d: np.ndarray, H: int) -> np.ndarray:
        x_1d = np.asarray(x_1d, dtype=float).ravel()
        point_forecast, _ = tfm.forecast([x_1d], freq=[0])
        pf = np.asarray(point_forecast[0], dtype=float)
        return pf[:H]
    return _forecast_daily


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load data (shared)
    S_b, S_d = load_series_q(CFG)
    if CFG.verbose:
        print(f"Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")
        print(f"Data (D): {S_d.index.min().date()} → {S_d.index.max().date()} | n={len(S_d)}")

    # 2) Model + adapter
    tfm = build_timesfm20(horizon_len=CFG.max_horizon)
    forecast_daily_fn = make_forecast_daily_fn(tfm)

    # 3) Walk-forward (shared) + metrics (shared)
    df_eval = walk_forward_q(S_b, S_d, CFG, forecast_daily_fn)
    eval_df = evaluate_q(df_eval)

    # 4) DM vs driftless RW at cut-level (shared)
    dm_against_rw_q(eval_df, loss="mse", h=1)

    # 5) Plot (shared)
    plot_q(
        eval_df,
        title="TimesFM 2.0 Forecast vs Actual (Quarterly Mean, EUR/NOK)",
        png_path=FIG_PNG,
        pdf_path=FIG_PDF,
        forecast_label="Forecast (TimesFM 2.0)",
    )


if __name__ == "__main__":
    main()
