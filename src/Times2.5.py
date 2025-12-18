# -*- coding: utf-8 -*-
"""
TimesFM 2.5 – EUR/NOK walk-forward (quarterly, levels) using eval_common_quarterly.py

- Shared evaluation ensures:
  - Same cut definition
  - Same quarterly target (mean over business days)
  - Same driftless RW benchmark for DM (must match eval_common_quarterly)
  - Same metrics + DM + plotting

Prereqs (example):
    pip install pandas numpy scikit-learn requests certifi matplotlib joblib
    pip install torch --index-url https://download.pytorch.org/whl/cpu  # or cuda
    git clone https://github.com/google-research/timesfm.git
    cd timesfm && pip install -e .[torch] && cd ..
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import torch
import timesfm  # TimesFM 2.5 repo install

from eval_common_quarterly import (
    EvalConfig,
    load_series,
    walk_forward_quarterly,
    evaluate_quarterly,
    dm_against_rw,
    plot_quarterly,
)

# -----------------------------
# Config (shared)
# -----------------------------
CFG = EvalConfig(
    url=(
        "https://raw.githubusercontent.com/bredeespelid/"
        "Data_MasterOppgave/refs/heads/main/Variables/All_Variables/variables_daily.csv"
    ),
    series="EUR_NOK",
    q_freq="Q-DEC",
    min_hist_days=40,
    max_context=2048,
    max_horizon=512,   # quarterly calendar days <= ~92, 512 is safe
    retries=3,
    timeout=60,
    verbose=True,
)

FIG_PNG = "EUR_NOK_TimesFM25_Q.png"
FIG_PDF = "EUR_NOK_TimesFM25_Q.pdf"


# -----------------------------
# Model (TimesFM 2.5 – 200M torch)
# -----------------------------
def build_timesfm25(max_context: int, horizon_len: int):
    """
    Returns a compiled TimesFM 2.5 model instance.
    """
    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise RuntimeError(
            "TimesFM 2.5 API not found. Install from repo:\n"
            "  git clone https://github.com/google-research/timesfm.git\n"
            "  cd timesfm\n"
            "  pip install -e .[torch]"
        )

    repo_id = "google/timesfm-2.5-200m-pytorch"
    cls = timesfm.TimesFM_2p5_200M_torch

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model = None
    if hasattr(cls, "from_pretrained"):
        try:
            if CFG.verbose:
                print(f"Loading TimesFM 2.5 checkpoint from Hugging Face: {repo_id}")
            model = cls.from_pretrained(repo_id, torch_compile=False)
        except Exception as e:
            if CFG.verbose:
                print(f"Could not load from HF: {e}. Falling back to local init.")

    if model is None:
        model = cls()
        if hasattr(model, "load_checkpoint"):
            try:
                model.load_checkpoint()
            except Exception as e:
                if CFG.verbose:
                    print(f"Warning: load_checkpoint failed ({e}); using current weights.")

    # Compile with ForecastConfig if available (repo API)
    if hasattr(timesfm, "ForecastConfig") and hasattr(model, "compile"):
        fc = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
        model.compile(fc)

    return model


def make_forecast_daily_fn(model):
    """
    Adapter to shared eval interface:
      forecast_daily_fn(x_1d, H) -> np.ndarray length H (daily point forecast)
    """
    def _forecast_daily(x_1d: np.ndarray, H: int) -> np.ndarray:
        if not hasattr(model, "forecast"):
            raise RuntimeError("TimesFM 2.5 model missing 'forecast' method.")
        out = model.forecast(horizon=H, inputs=[x_1d])
        if isinstance(out, tuple):
            # many versions return (point, quantiles)
            point = out[0][0]
        else:
            point = out[0]
        return np.asarray(point, dtype=float)[:H]
    return _forecast_daily


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Shared data loader
    S_b, S_d = load_series(CFG)

    # 2) Build model + adapter
    model = build_timesfm25(max_context=CFG.max_context, horizon_len=CFG.max_horizon)
    forecast_daily_fn = make_forecast_daily_fn(model)

    # 3) Shared walk-forward + metrics
    df_eval = walk_forward_quarterly(S_b, S_d, CFG, forecast_daily_fn)
    eval_df = evaluate_quarterly(df_eval)

    # 4) Shared DM vs RW (consistent benchmark)
    dm_against_rw(eval_df, loss="mse", h=1)

    # 5) Shared plot
    plot_quarterly(
        eval_df,
        title="TimesFM 2.5 Forecast vs Actual (Quarterly Mean, EUR/NOK)",
        png_path=FIG_PNG,
        pdf_path=FIG_PDF,
    )


if __name__ == "__main__":
    main()
