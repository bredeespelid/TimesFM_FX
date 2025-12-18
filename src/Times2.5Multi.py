# -*- coding: utf-8 -*-
"""
TimesFM 2.5 – Multi-FX walk-forward (quarterly, levels) using eval_common_quarterly

- Data: MultiFXData.csv (comma CSV, dot decimals)
- Cut = last B-day in previous quarter
- Forecast next quarter daily -> aggregate to quarterly mean over business days
- Per FX: Observations, RMSE, MAE, Directional Accuracy, DM vs RW (as defined in eval_common_quarterly)
- Outputs: metrics CSV (and optional plot if your eval_common has it)

Prereqs:
  pip install pandas numpy scikit-learn requests certifi matplotlib
  # TimesFM 2.5 from repo:
  #   git clone https://github.com/google-research/timesfm.git
  #   cd timesfm && pip install -e .[torch] && cd ..
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import requests, certifi

import torch
import timesfm

from eval_common_quarterly import (
    EvalConfig,
    to_business_and_daily,      # (df_d, col) -> (S_b, S_d)
    walk_forward_quarterly,
    evaluate_quarterly,
    dm_against_rw,
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class LocalConfig:
    url: str = "https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/refs/heads/main/EURNOK/MultiFXData.csv"
    include_fx: Optional[List[str]] = None   # e.g. ["EUR","USD","SEK","DKK","GBP"]
    metrics_csv: str = "FX_TimesFM25_metrics_quarterly.csv"
    retries: int = 3
    timeout: int = 60
    verbose: bool = True

LCFG = LocalConfig()

# Shared quarterly evaluation config (ONE source of truth)
ECFG = EvalConfig(
    url=LCFG.url,          # not used by eval_common loader here; we load ourselves
    series="(multi)",
    q_freq="Q-DEC",
    min_hist_days=40,
    max_context=2048,
    max_horizon=256,
    retries=LCFG.retries,
    timeout=LCFG.timeout,
    verbose=LCFG.verbose,
)

# -----------------------------
# Download & load MultiFX
# -----------------------------
def download_csv_text(url: str, retries: int, timeout: int) -> str:
    last_err = None
    for k in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, verify=certifi.where())
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if k < retries:
                pass
    raise RuntimeError(f"Download failed: {last_err}")

def load_multi_fx(url: str) -> pd.DataFrame:
    text = download_csv_text(url, LCFG.retries, LCFG.timeout)
    raw = pd.read_csv(io.StringIO(text), sep=",", encoding="utf-8-sig", decimal=".")
    if "DATE" not in raw.columns:
        raise ValueError(f"Expected DATE column; got: {list(raw.columns)[:10]} ...")

    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
    raw = raw.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")

    num_df = raw.apply(pd.to_numeric, errors="coerce")
    daily_idx = pd.date_range(num_df.index.min(), num_df.index.max(), freq="D")
    df_d = num_df.reindex(daily_idx).ffill()
    df_d.index.name = "DATE"
    return df_d

# -----------------------------
# TimesFM 2.5 model
# -----------------------------
def build_timesfm25(max_context: int, horizon_len: int):
    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise RuntimeError(
            "TimesFM 2.5 API not found. Install from repo with torch extras:\n"
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
            if LCFG.verbose:
                print(f"Loading TimesFM 2.5 checkpoint from Hugging Face: {repo_id}")
            model = cls.from_pretrained(repo_id, torch_compile=False)
        except Exception as e:
            if LCFG.verbose:
                print(f"Could not load from HF: {e}. Falling back to local init.")

    if model is None:
        model = cls()

    if hasattr(timesfm, "ForecastConfig") and hasattr(model, "compile"):
        fc = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
        model.compile(fc)

    return model

def make_forecast_daily_fn(model):
    def _forecast_daily(x_1d: np.ndarray, H: int) -> np.ndarray:
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
    df_d = load_multi_fx(LCFG.url)

    all_cols = [c for c in df_d.columns if pd.api.types.is_numeric_dtype(df_d[c])]
    fx_cols = [c for c in LCFG.include_fx if c in all_cols] if LCFG.include_fx else all_cols

    if LCFG.verbose:
        print(f"Running {len(fx_cols)} series: {fx_cols}")

    model = build_timesfm25(max_context=ECFG.max_context, horizon_len=min(ECFG.max_horizon, 256))
    forecast_daily_fn = make_forecast_daily_fn(model)

    metrics_rows: List[Dict] = []

    for col in fx_cols:
        # IMPORTANT: make S_b/S_d via eval_common to ensure identical definitions across scripts
        S_b, S_d = to_business_and_daily(df_d, col)

        df_eval = walk_forward_quarterly(S_b, S_d, ECFG, forecast_daily_fn, series_name=col)
        if df_eval.empty:
            if LCFG.verbose:
                print(f"[{col}] No evaluable quarters; skipping.")
            continue

        eval_df = evaluate_quarterly(df_eval, label=f"{col} (quarterly mean)")
        dm_stat, p_val = dm_against_rw(eval_df, loss="mse", h=1)

        # collect row
        row = {
            "series": col,
            "observations": int(len(eval_df)),
            "rmse": float(eval_df.attrs.get("rmse", np.nan)),
            "mae": float(eval_df.attrs.get("mae", np.nan)),
            "dir_acc": float(eval_df.attrs.get("dir_acc", np.nan)),
            "dm_stat": float(dm_stat) if np.isfinite(dm_stat) else np.nan,
            "dm_pvalue": float(p_val) if np.isfinite(p_val) else np.nan,
        }
        metrics_rows.append(row)

    if not metrics_rows:
        print("No series produced metrics. Check data and settings.")
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    metrics_df.to_csv(LCFG.metrics_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics to: {LCFG.metrics_csv}")

if __name__ == "__main__":
    main()
