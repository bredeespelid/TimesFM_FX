# -*- coding: utf-8 -*-
"""
TimesFM 2.5 – Multi-FX walk-forward (monthly, levels) using eval_common.py
- Data: MultiFXData.csv (daily, comma CSV)
- Cut = last B-day in previous month
- Forecast next month daily -> aggregate to monthly mean over B-days
- Evaluation/DM via eval_common (driftless RW, cut-level)
- Output: metrics CSV (one row per series)

Prereqs:
  pip install pandas numpy scikit-learn requests certifi matplotlib
  # TimesFM 2.5 (repo):
  #   git clone https://github.com/google-research/timesfm.git
  #   cd timesfm && pip install -e . && cd ..
"""

from __future__ import annotations
import io, time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, List

import numpy as np
import pandas as pd
import requests, certifi

import timesfm  # TimesFM 2.5 repo install required

from eval_common import (
    last_trading_day,
    period_range_from_series_index,
    business_days_in_period,
    calendar_days_in_period,
    aggregate_daily_to_bday_mean,
    evaluate_period_df,
)

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    url: str = "https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/refs/heads/main/EURNOK/MultiFXData.csv"
    freq: str = "M"
    min_hist_days: int = 40
    max_context: int = 2048
    max_horizon: int = 256
    retries: int = 3
    timeout: int = 60
    verbose: bool = True
    metrics_csv: str = "FX_TimesFM25_metrics_monthly.csv"
    include_fx: Optional[List[str]] = None
    dm_alpha: float = 0.10

CFG = Config()

# -----------------------------
# Download & data
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
                wait = 1.5 * k
                print(f"[warning] Download failed (try {k}/{retries}): {e}. Retrying in {wait:.1f}s ...")
                time.sleep(wait)
    raise RuntimeError(f"Download failed: {last_err}")

def load_multi_fx_daily(url: str) -> pd.DataFrame:
    text = download_csv_text(url, CFG.retries, CFG.timeout)
    # Try common CSV variants: comma, then semicolon with decimal comma, then semicolon with decimal dot
    variants = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
    ]
    raw = None
    last_err = None
    for v in variants:
        try:
            candidate = pd.read_csv(io.StringIO(text), sep=v["sep"], encoding="utf-8-sig", decimal=v["decimal"])
            if "DATE" in candidate.columns:
                raw = candidate
                break
        except Exception as e:
            last_err = e
            continue
    if raw is None:
        # Fallback: auto-scan first header to help debugging
        first_line = text.splitlines()[0] if text else "<empty>"
        raise ValueError(f"Could not parse CSV with expected DATE column. First header: {first_line}")

    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
    raw = raw.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")

    num_df = raw.apply(pd.to_numeric, errors="coerce")
    daily_idx = pd.date_range(num_df.index.min(), num_df.index.max(), freq="D")
    df_d = num_df.reindex(daily_idx).ffill()
    df_d.index.name = "DATE"
    return df_d

def series_daily_and_b(df_d: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    S_d = df_d[col].astype(float)
    S_d.name = col
    S_b = S_d.asfreq("B").ffill()
    S_b.name = col
    return S_b, S_d

# -----------------------------
# TimesFM 2.5 model
# -----------------------------
def build_timesfm25(max_context: int, horizon_len: int) -> Callable[[np.ndarray, int], np.ndarray]:
    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise RuntimeError("TimesFM 2.5 repo API not found. Install with: pip install -e . (from repo).")

    repo_id = "google/timesfm-2.5-200m-pytorch"
    cls = timesfm.TimesFM_2p5_200M_torch
    model = None

    if hasattr(cls, "from_pretrained"):
        try:
            if CFG.verbose:
                print(f"Loading TimesFM checkpoint from Hugging Face: {repo_id}")
            model = cls.from_pretrained(repo_id, torch_compile=False)
        except Exception as e:
            if CFG.verbose:
                print(f"Could not load from Hugging Face: {e}. Falling back to local init.")

    if model is None:
        model = cls()
        if hasattr(model, "load_checkpoint"):
            try:
                model.load_checkpoint()
            except Exception:
                if CFG.verbose:
                    print("Warning: could not load checkpoint; using current weights.")

    if hasattr(timesfm, "ForecastConfig") and hasattr(model, "compile"):
        cfg = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
        model.compile(cfg)

    def _forecast(x: np.ndarray, H: int) -> np.ndarray:
        out = model.forecast(horizon=H, inputs=[x])
        point = out[0][0] if isinstance(out, tuple) else out[0]
        return np.asarray(point, dtype=float)[:H]

    return _forecast

# -----------------------------
# Walk-forward (monthly)
# -----------------------------
def walk_forward_monthly(
    S_b: pd.Series,
    S_d: pd.Series,
    forecast_fn: Callable[[np.ndarray, int], np.ndarray],
    series_name: str,
) -> pd.DataFrame:
    months = period_range_from_series_index(S_b.index, CFG.freq)

    rows: Dict[str, dict] = {}
    dropped: Dict[str, str] = {}

    for m in months:
        prev_m = m - 1
        cut = last_trading_day(S_b, prev_m.start_time, prev_m.end_time)
        if cut is None:
            dropped[str(m)] = "no_cut_in_prev_month"
            continue

        hist_d = S_d.loc[:cut]
        if hist_d.size < CFG.min_hist_days:
            dropped[str(m)] = f"hist<{CFG.min_hist_days}"
            continue

        idx_m_b = business_days_in_period(S_b, m)
        if idx_m_b.size < 1:
            dropped[str(m)] = "no_bdays_in_month"
            continue

        y_true = float(S_b.loc[idx_m_b].mean())
        cut_level = float(S_b.loc[cut])  # <- for cut-level RW

        # Forecast horizon = full calendar days in the calendar month
        f_cal = calendar_days_in_period(m)
        H = len(f_cal)
        if H <= 0 or H > CFG.max_horizon:
            dropped[str(m)] = f"horizon_invalid(H={H})"
            continue

        context = min(CFG.max_context, len(hist_d))
        x = hist_d.values[-context:]

        pf = forecast_fn(x, H)
        if pf.shape[0] < H:
            dropped[str(m)] = f"horizon_short({pf.shape[0]})"
            continue

        pred_daily = pd.Series(pf[:H], index=f_cal, name="point")
        y_pred = aggregate_daily_to_bday_mean(pred_daily, idx_m_b)
        if not np.isfinite(y_pred):
            dropped[str(m)] = "no_overlap_pred_B_days"
            continue

        rows[str(m)] = {
            "series": series_name,
            "month": m,
            "cut": cut,
            "cut_level": cut_level,
            "y_true": y_true,
            "y_pred": float(y_pred),
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.set_index("month").sort_index()

    if CFG.verbose and dropped:
        miss = [str(mm) for mm in months if mm not in df.index]
        if miss:
            print(f"[{series_name}] Dropped months:")
            for mm in miss:
                print(f"  {mm}: {dropped.get(mm, 'unknown')}")

    return df

# -----------------------------
# Main
# -----------------------------
def main():
    df_d = load_multi_fx_daily(CFG.url)
    all_cols = [c for c in df_d.columns if pd.api.types.is_numeric_dtype(df_d[c])]

    fx_cols = [c for c in CFG.include_fx if c in all_cols] if CFG.include_fx else all_cols

    if CFG.verbose:
        print(f"Running monthly walk-forward for {len(fx_cols)} series.")

    forecast_fn = build_timesfm25(max_context=CFG.max_context, horizon_len=min(CFG.max_horizon, 256))

    metrics_rows = []
    for col in fx_cols:
        S_b, S_d = series_daily_and_b(df_d, col)
        if CFG.verbose:
            print(f"\n[{col}] Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")

        df_eval = walk_forward_monthly(S_b, S_d, forecast_fn, series_name=col)
        if df_eval.empty:
            if CFG.verbose:
                print(f"[{col}] No evaluable months; skipping.")
            continue

        _, res = evaluate_period_df(
            df_eval,
            loss="mse",
            h=1,
            alpha=CFG.dm_alpha,
            print_output=CFG.verbose,
            label=f"{col} (monthly mean)"
        )

        metrics_rows.append({
            "series": col,
            "observations": res.n_obs,
            "rmse": res.rmse,
            "mae": res.mae,
            "dir_hits": res.dir_hits,
            "dir_total": res.dir_total,
            "dir_acc": res.dir_acc,
            "dm_stat": res.dm_stat,
            "dm_pvalue": res.dm_pvalue,
            "better_than_rw": int(res.is_better_than_rw),
        })

    if not metrics_rows:
        print("No series produced metrics. Check data and settings.")
        return

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["rmse", "dm_pvalue"], ascending=[True, True])
    metrics_df.to_csv(CFG.metrics_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics to: {CFG.metrics_csv}")

if __name__ == "__main__":
    main()
