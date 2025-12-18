# -*- coding: utf-8 -*-
"""
Common evaluation utilities for EUR/NOK walk-forward (monthly).
- Same data loading (S_b business days, S_d daily)
- Same cut definition (last business day in prev month)
- Same target definition (monthly mean over business days)
- Same RW benchmark for DM (driftless: last observed level at cut)
- Same metrics + DM-test + plotting
"""

from __future__ import annotations

import io
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, Literal

import numpy as np
import pandas as pd
import requests
import certifi
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


LossType = Literal["mse", "mae"]
ForecastDailyFn = Callable[[np.ndarray, int], np.ndarray]  # (context_1d, H) -> daily point forecast length H


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class EvalConfig:
    url: str
    series: str = "EUR_NOK"
    m_freq: str = "M"
    min_hist_days: int = 40
    max_context: int = 2048
    max_horizon: int = 64
    retries: int = 3
    timeout: int = 60
    verbose: bool = True


# -----------------------------
# Download + Data
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


def load_series(cfg: EvalConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      S_b: business-day (B) series with ffill (for cuts and monthly truth)
      S_d: daily (D) series with ffill (for model context)
    """
    text = download_csv_text(cfg.url, cfg.retries, cfg.timeout)
    raw = pd.read_csv(io.StringIO(text))

    required_cols = {"Date", cfg.series}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Got: {list(raw.columns)}")

    df = (
        raw[["Date", cfg.series]]
        .rename(columns={"Date": "DATE"})
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"], errors="coerce"))
        .dropna(subset=["DATE", cfg.series])
        .sort_values("DATE")
        .set_index("DATE")
    )
    df[cfg.series] = pd.to_numeric(df[cfg.series], errors="coerce")
    df = df.dropna(subset=[cfg.series])

    S_b = df[cfg.series].asfreq("B").ffill().astype(float)
    S_b.name = cfg.series

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    S_d = df[cfg.series].reindex(full_idx).ffill().astype(float)
    S_d.index.name = "DATE"
    S_d.name = cfg.series
    return S_b, S_d


def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    sl = S_b.loc[start:end]
    return None if sl.empty else sl.index[-1]


# -----------------------------
# Walk-forward (monthly) generic runner
# -----------------------------
def walk_forward_monthly(
    S_b: pd.Series,
    S_d: pd.Series,
    cfg: EvalConfig,
    forecast_daily_fn: ForecastDailyFn,
) -> pd.DataFrame:
    """
    Generic monthly walk-forward:
      - cut = last B-day of prev month
      - y_true = mean of S_b in month m (business days)
      - y_pred = mean of predicted daily values on business days in month m
      - rw_pred = driftless RW level at cut (S_b.loc[cut]) for consistent DM
    """
    first_m = pd.Period(S_b.index.min(), freq=cfg.m_freq)
    last_m = pd.Period(S_b.index.max(), freq=cfg.m_freq)
    months = pd.period_range(first_m, last_m, freq=cfg.m_freq)

    rows: Dict[str, Dict] = {}
    dropped: Dict[str, str] = {}

    for m in months:
        prev_m = m - 1
        m_start, m_end = m.start_time, m.end_time
        prev_start, prev_end = prev_m.start_time, prev_m.end_time

        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(m)] = "no_cut_in_prev_month"
            continue

        hist_d = S_d.loc[:cut]
        if hist_d.size < cfg.min_hist_days:
            dropped[str(m)] = f"hist<{cfg.min_hist_days}"
            continue

        idx_m_b = S_b.index[(S_b.index >= m_start) & (S_b.index <= m_end)]
        if idx_m_b.size < 1:
            dropped[str(m)] = "no_bdays_in_month"
            continue

        y_true = float(S_b.loc[idx_m_b].mean())
        rw_pred = float(S_b.loc[cut])  # driftless RW benchmark (last observed level at cut)

        H = (m_end.date() - m_start.date()).days + 1
        if H <= 0 or H > cfg.max_horizon:
            dropped[str(m)] = f"horizon_invalid(H={H})"
            continue

        context = min(cfg.max_context, len(hist_d))
        x = hist_d.values[-context:]

        pf = np.asarray(forecast_daily_fn(x, H), dtype=float)
        if pf.shape[0] < H:
            dropped[str(m)] = f"horizon_short({pf.shape[0]})"
            continue
        pf = pf[:H]

        f_idx = pd.date_range(cut + pd.Timedelta(days=1), periods=H, freq="D")
        pred_daily = pd.Series(pf, index=f_idx, name="point")

        pred_b = pred_daily.reindex(idx_m_b, method=None)
        if pred_b.isna().all():
            dropped[str(m)] = "no_overlap_pred_B_days"
            continue

        y_pred = float(pred_b.dropna().mean())

        rows[str(m)] = {"month": m, "cut": cut, "y_true": y_true, "y_pred": y_pred, "rw_pred": rw_pred}

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.set_index("month").sort_index()

    if cfg.verbose and dropped:
        miss = [str(m) for m in months if m not in df.index]
        if miss:
            print("\nDropped months and reasons:")
            for mm in miss:
                print(f"  {mm}: {dropped.get(mm, 'unknown')}")
    return df


# -----------------------------
# Metrics + DM
# -----------------------------
def evaluate_monthly(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = eval_df.copy()
    df["err"] = df["y_true"] - df["y_pred"]
    core = df.dropna(subset=["y_true", "y_pred"]).copy()

    n_obs = int(len(core))
    rmse = float(np.sqrt(np.mean(np.square(core["err"])))) if n_obs else np.nan
    mae = float(mean_absolute_error(core["y_true"], core["y_pred"])) if n_obs else np.nan

    core["y_prev"] = core["y_true"].shift(1)
    mask = core["y_prev"].notna()
    dir_true = np.sign(core.loc[mask, "y_true"] - core.loc[mask, "y_prev"])
    dir_pred = np.sign(core.loc[mask, "y_pred"] - core.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    hit_rate = (hits / total) if total else np.nan

    print("\n=== Model performance (monthly mean, EUR/NOK) ===")
    print(f"Observations: {n_obs}")
    print(f"RMSE (level): {rmse:.6f}")
    print(f"MAE  (level): {mae:.6f}")
    if total:
        print(f"Directional accuracy: {hits}/{total} ({hit_rate*100:.1f}%)")

    return core


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def dm_test(y_true: pd.Series, y_model: pd.Series, y_rw: pd.Series, h: int = 1, loss: LossType = "mse") -> Tuple[float, float]:
    df = pd.concat({"y": y_true, "m": y_model, "rw": y_rw}, axis=1).dropna()
    if df.empty or len(df) < 5:
        return float("nan"), float("nan")

    e_m = df["y"] - df["m"]
    e_r = df["y"] - df["rw"]
    d = np.abs(e_m) - np.abs(e_r) if loss.lower() == "mae" else (e_m ** 2) - (e_r ** 2)

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


def dm_against_rw(eval_df: pd.DataFrame, loss: LossType = "mse", h: int = 1) -> None:
    if "rw_pred" not in eval_df.columns:
        raise ValueError("Missing 'rw_pred'. Ensure walk_forward_monthly() output is used.")
    dm_stat, p_val = dm_test(eval_df["y_true"], eval_df["y_pred"], eval_df["rw_pred"], h=h, loss=loss)
    print("\n=== Diebold–Mariano vs Random Walk (driftless, cut-level) ===")
    print(f"Loss: {loss.upper()} | horizon h={h}")
    print(f"DM-statistic: {dm_stat:.4f}" if np.isfinite(dm_stat) else "DM-statistic: nan")
    print(f"p-value     : {p_val:.4f}" if np.isfinite(p_val) else "p-value     : nan")


# -----------------------------
# Plot
# -----------------------------
def plot_monthly(eval_df: pd.DataFrame, title: str, png_path: str, pdf_path: str) -> None:
    if eval_df.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    plt.plot(x, eval_df["y_true"], color="black", label="Actual (monthly mean, B-days)")
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--", label="Forecast")

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
