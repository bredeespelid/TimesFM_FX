# -*- coding: utf-8 -*-
"""
TimesFM 2.5 – Multi-FX walk-forward (monthly, levels) with unified metrics
- Data: MultiFXData.csv (comma CSV, dot decimals) at:
  https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/refs/heads/main/EURNOK/MultiFXData.csv
- Cut = last B-day in previous month
- Forecast next month at daily frequency -> aggregate to monthly mean over business days
- Per FX: Observations, RMSE, MAE, Directional Accuracy, DM test vs Random Walk (MSE, h=1)
- Outputs: metrics CSV (one row per series)

Prereqs:
  pip install pandas numpy scikit-learn requests certifi matplotlib
  # TimesFM 2.5 (from repo):
  #   git clone https://github.com/google-research/timesfm.git
  #   cd timesfm && pip install -e . && cd ..
"""

from __future__ import annotations
import io, time, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, List

import numpy as np
import pandas as pd
import requests, certifi
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import timesfm  # TimesFM 2.5 repo install required

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    url: str = "https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/refs/heads/main/EURNOK/MultiFXData.csv"
    m_freq: str = "M"           # monthly periods, month-end
    min_hist_days: int = 40
    max_context: int = 2048
    max_horizon: int = 256
    retries: int = 3
    timeout: int = 60
    verbose: bool = True
    metrics_csv: str = "FX_TimesFM25_metrics_monthly.csv"
    include_fx: Optional[List[str]] = None  # e.g. ["EUR","USD","SEK","DKK","GBP"]

CFG = Config()

# -----------------------------
# Download & data prep
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

def load_multi_fx(url: str) -> pd.DataFrame:
    """
    Reads MultiFXData.csv with columns like:
      DATE, I44, AUD, CHF, DKK, EUR, CAD, GBP, ..., USD, ...

    Be tolerant to delimiter/style differences:
      - Prefer comma separator; if no DATE column found, retry with semicolon.
      - Prefer dot decimals; if too many NaNs after parsing, retry with decimal=",".

    Returns daily DataFrame indexed by DATE with ffilled numeric series.
    """
    text = download_csv_text(url, CFG.retries, CFG.timeout)

    def _try_read(sep: str, decimal: str) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(text), sep=sep, encoding="utf-8-sig", decimal=decimal)

    # 1) Try comma + dot (typical)
    raw = _try_read(",", ".")
    if "DATE" not in raw.columns:
        # 2) Fallback: semicolon + dot
        raw = _try_read(";", ".")
    if "DATE" not in raw.columns:
        # 3) Last resort: comma + comma-decimal, then semicolon + comma-decimal
        for sep in (",", ";"):
            raw = _try_read(sep, ",")
            if "DATE" in raw.columns:
                break
    if "DATE" not in raw.columns:
        raise ValueError(f"Expected a DATE column; got: {list(raw.columns)[:10]} ...")

    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
    raw = raw.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")

    num_df = raw.apply(pd.to_numeric, errors="coerce")
    daily_idx = pd.date_range(num_df.index.min(), num_df.index.max(), freq="D")
    df_d = num_df.reindex(daily_idx).ffill()
    df_d.index.name = "DATE"
    return df_d

def series_daily_and_b(df_d: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    """
    For one FX column -> returns (S_b, S_d).
    """
    if col not in df_d.columns:
        raise ValueError(f"Column {col} not found.")
    S_d = df_d[col].astype(float)
    S_d.name = col
    S_b = S_d.asfreq("B").ffill()
    S_b.name = col
    return S_b, S_d

def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    sl = S_b.loc[start:end]
    return sl.index[-1] if not sl.empty else None

# -----------------------------
# Model (TimesFM 2.5 only)
# -----------------------------
def build_model(max_context: int, horizon_len: int) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Returns forecast_fn(x, H) -> np.ndarray length H (point forecast).
    Requires TimesFM 2.5 repo API.
    """
    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise RuntimeError("TimesFM 2.5 repo API not found. Ensure the 2.5 package is installed (pip install -e .).")

    repo_id = "google/timesfm-2.5-200m-pytorch"
    model = None
    cls = timesfm.TimesFM_2p5_200M_torch

    if hasattr(cls, "from_pretrained"):
        try:
            if CFG.verbose:
                print(f"Loading TimesFM checkpoint from Hugging Face: {repo_id}")
            model = cls.from_pretrained(repo_id, torch_compile=False)
        except Exception as e:
            if CFG.verbose:
                print(f"Could not load checkpoint from Hugging Face: {e}. Falling back to local init.")

    if model is None:
        model = cls()
        if hasattr(model, "load_checkpoint"):
            try:
                model.load_checkpoint()
            except TypeError:
                try:
                    model.load_checkpoint(path=None)
                except (TypeError, NotImplementedError):
                    if CFG.verbose:
                        print("Warning: load_checkpoint not available; using randomly init weights (not recommended).")

    if hasattr(timesfm, "ForecastConfig"):
        cfg = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,  # point forecasts only
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
        if hasattr(model, "compile"):
            model.compile(cfg)

    def _forecast(x: np.ndarray, H: int) -> np.ndarray:
        if not hasattr(model, "forecast"):
            raise RuntimeError("TimesFM 2.5 API missing 'forecast'.")
        out = model.forecast(horizon=H, inputs=[x])
        if isinstance(out, tuple):  # (point, quantiles)
            point = out[0][0]
        else:
            point = out[0]
        return np.asarray(point, dtype=float)[:H]

    return _forecast

# -----------------------------
# Walk-forward (monthly) – point forecasts only
# -----------------------------
def walk_forward_monthly(
    S_b: pd.Series,
    S_d: pd.Series,
    forecast_fn: Callable[[np.ndarray, int], np.ndarray],
    series_name: str,
) -> pd.DataFrame:
    """
    Monthly walk-forward:
      - Cut = last B-day in previous month
      - Forecast next month at daily frequency
      - Aggregate to monthly mean over business days
    """
    first_m = pd.Period(S_b.index.min(), freq=CFG.m_freq)
    last_m  = pd.Period(S_b.index.max(),  freq=CFG.m_freq)
    months = pd.period_range(first_m, last_m, freq=CFG.m_freq)

    rows: Dict = {}
    dropped: Dict[str, str] = {}

    for m in months:
        prev_m = m - 1
        m_start, m_end = m.start_time, m.end_time
        prev_start, prev_end = prev_m.start_time, prev_m.end_time

        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(m)] = "no_cut_in_prev_m"
            continue

        hist_d = S_d.loc[:cut]
        if hist_d.size < CFG.min_hist_days:
            dropped[str(m)] = f"hist<{CFG.min_hist_days}"
            continue

        idx_m_b = S_b.index[(S_b.index >= m_start) & (S_b.index <= m_end)]
        if idx_m_b.size < 1:
            dropped[str(m)] = "no_bdays_in_m"
            continue
        y_true = float(S_b.loc[idx_m_b].mean())

        H = (m_end.date() - m_start.date()).days + 1
        if H <= 0 or H > CFG.max_horizon:
            dropped[str(m)] = f"horizon_invalid(H={H})"
            continue

        context = min(CFG.max_context, len(hist_d))
        x = hist_d.values[-context:]

        pf = forecast_fn(x, H)
        if pf.shape[0] < H:
            dropped[str(m)] = f"horizon_short({pf.shape[0]})"
            continue

        f_idx = pd.date_range(cut + pd.Timedelta(days=1), periods=H, freq="D")
        pred_daily = pd.Series(pf[:H], index=f_idx, name="point")

        pred_b = pred_daily.reindex(idx_m_b, method=None)
        if pred_b.isna().all():
            dropped[str(m)] = "no_overlap_pred_B_days"
            continue
        y_pred = float(pred_b.dropna().mean())

        rows[str(m)] = {
            "series": series_name,
            "month": m,
            "cut": cut,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.set_index("month").sort_index()

    if CFG.verbose and dropped:
        miss = [str(m) for m in months if m not in df.index]
        if miss:
            print(f"[{series_name}] Dropped months:")
            for mm in miss:
                print(f"  {mm}: {dropped.get(mm, 'unknown')}")

    return df

# -----------------------------
# Evaluation & DM
# -----------------------------
def evaluate(eval_df: pd.DataFrame) -> Dict[str, float]:
    df = eval_df.copy()
    df["err"] = df["y_true"] - df["y_pred"]
    core = df.dropna(subset=["y_true", "y_pred"]).copy()

    n_obs = int(len(core))
    rmse = float(np.sqrt(np.mean(np.square(core["err"])))) if n_obs else np.nan
    mae  = float(mean_absolute_error(core["y_true"], core["y_pred"])) if n_obs else np.nan

    core["y_prev"] = core["y_true"].shift(1)
    mask = core["y_prev"].notna()
    dir_true = np.sign(core.loc[mask, "y_true"] - core.loc[mask, "y_prev"])
    dir_pred = np.sign(core.loc[mask, "y_pred"] - core.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    hit_rate = (hits / total) if total else np.nan

    if CFG.verbose:
        if total:
            print(
                f"Observations: {n_obs} | RMSE={rmse:.6f} | MAE={mae:.6f} | "
                f"DirAcc={hits}/{total} ({hit_rate*100:.1f}%)"
            )
        else:
            print(
                f"Observations: {n_obs} | RMSE={rmse:.6f} | MAE={mae:.6f} | DirAcc=NA"
            )

    return {
        "observations": n_obs,
        "rmse": rmse,
        "mae": mae,
        "dir_hits": hits,
        "dir_total": total,
        "dir_acc": hit_rate if total else np.nan,
    }

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def dm_test(y_true: pd.Series, y_model: pd.Series, y_rw: pd.Series, h: int = 1, loss: str = "mse"):
    df = pd.concat({"y": y_true, "m": y_model, "rw": y_rw}, axis=1).dropna()
    if df.empty or len(df) < 5:
        return float("nan"), float("nan")
    e_m = df["y"] - df["m"]
    e_r = df["y"] - df["rw"]
    d = np.abs(e_m) - np.abs(e_r) if loss.lower() == "mae" else (e_m**2) - (e_r**2)
    N = int(len(d))
    d_mean = float(d.mean())
    gamma0 = float(np.var(d, ddof=1)) if N > 1 else 0.0
    var_bar = gamma0 / N
    if h > 1 and N > 2:
        for k in range(1, min(h - 1, N - 1) + 1):
            w_k = 1.0 - k / h
            cov_k = float(np.cov(d[k:], d[:-k], ddof=1)[0, 1])
            var_bar += 2.0 * w_k * cov_k / N
    if var_bar <= 0 or not np.isfinite(var_bar):
        return float("nan"), float("nan")
    dm_stat = d_mean / math.sqrt(var_bar)
    p_val = 2.0 * (1.0 - _normal_cdf(abs(dm_stat)))
    return dm_stat, p_val

def evaluate_with_dm(eval_df: pd.DataFrame) -> Dict[str, float]:
    m = evaluate(eval_df)
    df = eval_df.copy()
    df["rw_pred"] = df["y_true"].shift(1)
    dm_stat, p_val = dm_test(df["y_true"], df["y_pred"], df["rw_pred"], h=1, loss="mse")
    m["dm_stat"] = float(dm_stat) if np.isfinite(dm_stat) else np.nan
    m["dm_pvalue"] = float(p_val) if np.isfinite(p_val) else np.nan
    return m

# -----------------------------
# Main
# -----------------------------
def main():
    # Load all FX daily frame
    df_d = load_multi_fx(CFG.url)
    # Determine which columns to run (exclude non-numeric automatically)
    all_cols = [c for c in df_d.columns if pd.api.types.is_numeric_dtype(df_d[c])]
    if CFG.include_fx:
        fx_cols = [c for c in CFG.include_fx if c in all_cols]
    else:
        fx_cols = all_cols  # includes I44, TWI, XDR, etc., if present

    if CFG.verbose:
        print(f"Running monthly walk-forward for {len(fx_cols)} series:", fx_cols)

    # Build model once; reuse for all series
    forecast_fn = build_model(max_context=CFG.max_context, horizon_len=min(CFG.max_horizon, 256))

    metrics_rows = []
    for col in fx_cols:
        S_b, S_d = series_daily_and_b(df_d, col)
        if CFG.verbose:
            print(f"\n[{col}] Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")

        df_eval = walk_forward_monthly(S_b, S_d, forecast_fn, series_name=col)
        if df_eval.empty or df_eval["y_pred"].isna().all():
            if CFG.verbose:
                print(f"[{col}] No evaluable months; skipping.")
            continue

        m = evaluate_with_dm(df_eval)
        m["series"] = col
        metrics_rows.append(m)

        # Console summary per series
        if np.isfinite(m["dir_acc"]) and m["dir_total"] > 0:
            print(
                f"[{col}] Obs={m['observations']}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, "
                f"DirAcc={m['dir_hits']}/{m['dir_total']} ({m['dir_acc']*100:.1f}%), "
                f"DM={m['dm_stat']:.3f}, p={m['dm_pvalue']:.4f}"
            )
        else:
            print(
                f"[{col}] Obs={m['observations']}, RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, "
                f"DirAcc=NA, DM={m['dm_stat']:.3f}, p={m['dm_pvalue']:.4f}"
            )

    if not metrics_rows:
        print("No series produced metrics. Check data and settings.")
        return

    metrics_df = pd.DataFrame(metrics_rows)[
        ["series", "observations", "rmse", "mae", "dir_hits", "dir_total", "dir_acc", "dm_stat", "dm_pvalue"]
    ].sort_values("rmse")
    metrics_df.to_csv(CFG.metrics_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics to: {CFG.metrics_csv}")

if __name__ == "__main__":
    main()
