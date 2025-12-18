# -*- coding: utf-8 -*-
"""
eval_common.py
Unified evaluation utilities for walk-forward forecasting on period means (B-day mean),
with driftless Random Walk benchmark defined as CUT-LEVEL (y at cut).

Supports:
- monthly: freq="M"
- quarterly: freq="Q-DEC" (or any pandas Period freq)

Key conventions:
- Truth y_true(period) = mean of target over BUSINESS DAYS within the period.
- Forecasts are produced at DAILY frequency for the full calendar period, then
  aggregated to BUSINESS-DAY mean to match y_true.
- RW benchmark is driftless cut-level: rw_pred(period) = y(cut).
- DM test is two-sided; interpret "better than RW" as DM < 0 AND p < alpha.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


LossType = Literal["mse", "mae"]


@dataclass
class EvalResult:
    n_obs: int
    rmse: float
    mae: float
    dir_hits: int
    dir_total: int
    dir_acc: float
    dm_stat: float
    dm_pvalue: float
    is_better_than_rw: bool
    alpha: float


def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Last business day in [start, end] using an existing B-day series index."""
    sl = S_b.loc[start:end]
    return sl.index[-1] if not sl.empty else None


def period_range_from_series_index(idx: pd.DatetimeIndex, freq: str) -> pd.PeriodIndex:
    first_p = pd.Period(idx.min(), freq=freq)
    last_p  = pd.Period(idx.max(), freq=freq)
    return pd.period_range(first_p, last_p, freq=freq)


def business_days_in_period(S_b: pd.Series, p: pd.Period) -> pd.DatetimeIndex:
    """Business-day index within period p using S_b index."""
    start, end = p.start_time, p.end_time
    return S_b.index[(S_b.index >= start) & (S_b.index <= end)]


def calendar_days_in_period(p: pd.Period) -> pd.DatetimeIndex:
    """Calendar daily index from period start to end inclusive."""
    return pd.date_range(p.start_time, p.end_time, freq="D")


def aggregate_daily_to_bday_mean(pred_daily: pd.Series, idx_b: pd.DatetimeIndex) -> float:
    """Aggregate daily forecast to mean over business days within idx_b."""
    pred_b = pred_daily.reindex(idx_b, method=None)
    if pred_b.isna().all():
        return float("nan")
    return float(pred_b.dropna().mean())


def compute_directional_accuracy(core: pd.DataFrame, y_col: str = "y_true", pred_col: str = "y_pred") -> Tuple[int, int, float]:
    """
    Directional accuracy computed as hit-rate for sign(Δy_true) vs sign(Δy_pred),
    where Δ is relative to previous period's y_true.
    """
    tmp = core.copy()
    tmp["y_prev"] = tmp[y_col].shift(1)
    mask = tmp["y_prev"].notna()
    if mask.sum() == 0:
        return 0, 0, float("nan")

    dir_true = np.sign(tmp.loc[mask, y_col] - tmp.loc[mask, "y_prev"])
    dir_pred = np.sign(tmp.loc[mask, pred_col] - tmp.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    acc = hits / total if total else float("nan")
    return hits, total, acc


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def dm_test(
    y_true: pd.Series,
    y_model: pd.Series,
    y_rw: pd.Series,
    h: int = 1,
    loss: LossType = "mse",
) -> Tuple[float, float]:
    """
    Diebold–Mariano test (two-sided) with simple NW/Bartlett HAC up to lag h-1.

    d_t = L(e_model) - L(e_rw)
    DM < 0 => model has lower mean loss than RW.
    """
    df = pd.concat({"y": y_true, "m": y_model, "rw": y_rw}, axis=1).dropna()
    if df.empty or len(df) < 5:
        return float("nan"), float("nan")

    e_m = df["y"] - df["m"]
    e_r = df["y"] - df["rw"]

    if loss == "mae":
        d = np.abs(e_m) - np.abs(e_r)
    else:  # mse
        d = (e_m ** 2) - (e_r ** 2)

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
    return float(dm_stat), float(p_val)


def evaluate_period_df(
    eval_df: pd.DataFrame,
    *,
    loss: LossType = "mse",
    h: int = 1,
    alpha: float = 0.10,
    print_output: bool = True,
    label: str = "",
) -> Tuple[pd.DataFrame, EvalResult]:
    """
    eval_df must contain columns: y_true, y_pred, cut, and be indexed by PeriodIndex.
    Computes RMSE/MAE on levels (period means), directional accuracy, and DM vs cut-level RW.
    """
    df = eval_df.copy()
    core = df.dropna(subset=["y_true", "y_pred"]).copy()
    n_obs = int(len(core))

    if n_obs == 0:
        res = EvalResult(
            n_obs=0, rmse=float("nan"), mae=float("nan"),
            dir_hits=0, dir_total=0, dir_acc=float("nan"),
            dm_stat=float("nan"), dm_pvalue=float("nan"),
            is_better_than_rw=False, alpha=alpha
        )
        return core, res

    err = core["y_true"] - core["y_pred"]
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(mean_absolute_error(core["y_true"], core["y_pred"]))

    hits, total, dir_acc = compute_directional_accuracy(core, "y_true", "y_pred")

    # Driftless RW (cut-level): rw_pred(period) = y(cut)
    # Requires that 'cut' is a Timestamp and that the cut level is available via 'cut_level'.
    # If caller did not provide cut_level, fall back to NaN.
    if "cut_level" in core.columns:
        rw_pred = core["cut_level"]
    else:
        rw_pred = pd.Series(index=core.index, dtype=float)

    dm_stat, p_val = dm_test(core["y_true"], core["y_pred"], rw_pred, h=h, loss=loss)

    is_better = (np.isfinite(dm_stat) and np.isfinite(p_val) and (dm_stat < 0) and (p_val < alpha))

    if print_output:
        if label:
            print(f"\n=== {label} ===")
        print(f"Observations: {n_obs}")
        print(f"RMSE (level): {rmse:.6f}")
        print(f"MAE  (level): {mae:.6f}")
        if total:
            print(f"Directional accuracy: {hits}/{total} ({dir_acc*100:.1f}%)")

        print("\n=== Diebold–Mariano vs Random Walk (driftless, cut-level) ===")
        print(f"Loss: {loss.upper()} | horizon h={h}")
        print(f"DM-statistic: {dm_stat:.4f}" if np.isfinite(dm_stat) else "DM-statistic: nan")
        print(f"p-value     : {p_val:.4f}" if np.isfinite(p_val) else "p-value     : nan")
        if np.isfinite(dm_stat) and np.isfinite(p_val):
            print(f"Better than RW @ alpha={alpha:.2f}: {'YES' if is_better else 'NO'} (requires DM<0 and p<alpha)")

    res = EvalResult(
        n_obs=n_obs, rmse=rmse, mae=mae,
        dir_hits=hits, dir_total=total, dir_acc=dir_acc,
        dm_stat=dm_stat, dm_pvalue=p_val,
        is_better_than_rw=is_better, alpha=alpha
    )
    return core, res


def plot_period_simple(
    eval_df: pd.DataFrame,
    *,
    title: str,
    png_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    y_label: str = "Level",
    show: bool = True,
):
    """Simple line plot: actual vs forecast on period means."""
    if eval_df.empty:
        print("Nothing to plot.")
        return

    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    plt.figure(figsize=(10, 6))
    plt.plot(x, eval_df["y_true"], color="black", label="Actual (period mean, B-days)")
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--", label="Forecast")

    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if png_path:
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")
    if pdf_path:
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if show:
        plt.show()
