# -*- coding: utf-8 -*-
"""
TimesFM 2.5 (200M, PyTorch) – EUR/NOK walk-forward (monthly) using shared evaluation.

Shared evaluation (eval_common.py) ensures:
- Same cut definition
- Same monthly target (mean over business days)
- Same driftless RW benchmark for DM: rw_pred = S_b.loc[cut]
- Same metrics + DM + plotting
"""

from __future__ import annotations

import numpy as np
import torch
import timesfm  # TimesFM 2.5 API (torch)
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

FIG_PNG = "data/Times2.5M.png"
FIG_PDF = "data/Times2.5M.pdf"


# -----------------------------
# Model (TimesFM 2.5 – 200M, PyTorch)
# -----------------------------
def build_timesfm25_forecast_fn(max_context: int, horizon_len: int):
    """
    Initialize TimesFM 2.5 (200M, torch) and return a forecast function:
        forecast_daily_fn(x_1d, H) -> np.ndarray of length H (daily point forecast)
    """
    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise RuntimeError(
            "TimesFM 2.5 API not found. Install from repo with torch extras:\n"
            "  git clone https://github.com/google-research/timesfm.git\n"
            "  cd timesfm\n"
            "  pip install -e .[torch]"
        )

    repo_id = "google/timesfm-2.5-200m-pytorch"
    cls = timesfm.TimesFM_2p5_200M_torch
    model = None

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if hasattr(cls, "from_pretrained"):
        try:
            if CFG.verbose:
                print(f"Loading TimesFM 2.5 checkpoint from Hugging Face: {repo_id}")
            model = cls.from_pretrained(repo_id, torch_compile=False)
        except Exception as e:
            if CFG.verbose:
                print(f"Could not load checkpoint from Hugging Face: {e}. Falling back to local init.")

    if model is None:
        model = cls()
        if hasattr(model, "load_checkpoint"):
            try:
                model.load_checkpoint()
            except Exception:
                if CFG.verbose:
                    print("Warning: load_checkpoint failed/unavailable; using randomly initialized weights (not recommended).")

    if hasattr(timesfm, "ForecastConfig"):
        fc = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
        if hasattr(model, "compile"):
            model.compile(fc)

    def _forecast_daily(x_1d: np.ndarray, H: int) -> np.ndarray:
        if not hasattr(model, "forecast"):
            raise RuntimeError("TimesFM 2.5 model missing 'forecast' method.")
        out = model.forecast(horizon=H, inputs=[x_1d])
        if isinstance(out, tuple):
            point = out[0][0]
        else:
            point = out[0]
        return np.asarray(point, dtype=float)[:H]

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

    # 2) Forecast function for shared walk-forward
    forecast_daily_fn = build_timesfm25_forecast_fn(
        max_context=CFG.max_context,
        horizon_len=CFG.max_horizon,
    )

    # 3) Walk-forward (shared) + metrics (shared)
    df_eval = walk_forward_monthly(S_b, S_d, CFG, forecast_daily_fn)
    eval_df = evaluate_monthly(df_eval)

    # 4) DM vs driftless RW (shared, consistent)
    dm_against_rw(eval_df, loss="mse", h=1)

    # 5) Plot (shared)
    plot_monthly(
        eval_df,
        title="TimesFM 2.5 Forecast vs Actual (Monthly Mean, EUR/NOK)",
        png_path=FIG_PNG,
        pdf_path=FIG_PDF,
    )


if __name__ == "__main__":
    main()
