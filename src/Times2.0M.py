# -*- coding: utf-8 -*-
"""
TimesFM 2.0 (500M, PyTorch) – EUR/NOK walk-forward (monthly) without prediction intervals.
Source: GitHub all-variables daily panel (comma-separated).
"""

from __future__ import annotations
import io, time, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests, certifi
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt

import timesfm  # v1-API (timesfm==1.3.0)

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    url: str = (
        "https://raw.githubusercontent.com/bredeespelid/"
        "Data_MasterOppgave/refs/heads/main/Variables/All_Variables/variables_daily.csv"
    )
    m_freq: str = "M"      # monthly evaluation
    min_hist_days: int = 40
    max_context: int = 2048
    max_horizon: int = 64  # must exceed the longest month
    retries: int = 3
    timeout: int = 60
    verbose: bool = True
    fig_png: str = "EUR_NOK_TimesFM_vs_Actual_Monthly.png"
    fig_pdf: str = "EUR_NOK_TimesFM_vs_Actual_Monthly.pdf"

CFG = Config()

# -----------------------------
# Helper: download
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

# -----------------------------
# Data
# -----------------------------
def load_series(url: str) -> Tuple[pd.Series, pd.Series]:
    """
    Read the all-variables daily CSV from GitHub.

    Expected columns (at minimum):
      Date, EUR_NOK, ...

    Returns:
      S_b: business-day (B) EUR/NOK with forward fill (used for cuts and monthly truth)
      S_d: daily (D) EUR/NOK with forward fill (model inputs and daily forecasts)
    """
    text = download_csv_text(url, CFG.retries, CFG.timeout)
    raw = pd.read_csv(io.StringIO(text))

    required_cols = {"Date", "EUR_NOK"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Got: {list(raw.columns)}")

    df = (
        raw[list(required_cols)]
        .rename(columns={"Date": "DATE"})
        .assign(DATE=lambda x: pd.to_datetime(x["DATE"], errors="coerce"))
        .dropna(subset=["DATE", "EUR_NOK"])
        .sort_values("DATE")
        .set_index("DATE")
    )

    # Business-day series (truth / aggregation base)
    S_b = df["EUR_NOK"].asfreq("B").ffill().astype(float)
    S_b.name = "EUR_NOK"

    # Daily series (model inputs / forecasts)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    S_d = df["EUR_NOK"].reindex(full_idx).ffill().astype(float)
    S_d.index.name = "DATE"
    S_d.name = "EUR_NOK"
    return S_b, S_d

def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the last available business day in [start, end]."""
    sl = S_b.loc[start:end]
    if sl.empty:
        return None
    return sl.index[-1]

# -----------------------------
# Model (TimesFM 2.0 – 500M, PyTorch)
# -----------------------------
def build_model(max_context: int, horizon_len: int):
    """
    Build a TimesFM 2.0 (500M) model using the v1 PyTorch backend.
    """
    required = all([
        hasattr(timesfm, "TimesFm"),
        hasattr(timesfm, "TimesFmHparams"),
        hasattr(timesfm, "TimesFmCheckpoint"),
    ])
    if not required:
        raise RuntimeError(
            "TimesFM v1 API not found. Install 'timesfm==1.3.0' for 2.0-500M: "
            "python -m pip install timesfm==1.3.0"
        )

    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=32,
            horizon_len=horizon_len,   # must cover the longest month (~31); 64 is a safe margin
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

# -----------------------------
# Walk-forward (monthly) – point forecast only
# -----------------------------
def walk_forward_timesfm_monthly(S_b: pd.Series, S_d: pd.Series, tfm) -> pd.DataFrame:
    """
    Monthly walk-forward:

      - Cut at the last business day of the previous month (from S_b).
      - Use daily EUR/NOK history up to and including the cut (S_d).
      - Forecast the full next calendar month at daily frequency.
      - Aggregate daily forecasts to the monthly mean over business days.
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

        # Cut date for history
        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(m)] = "no_cut_in_prev_month"
            continue

        # Daily history up to cut
        hist_d = S_d.loc[:cut]
        if hist_d.size < CFG.min_hist_days:
            dropped[str(m)] = f"hist<{CFG.min_hist_days}"
            continue

        # Business days in the target month (truth)
        idx_m_b = S_b.index[(S_b.index >= m_start) & (S_b.index <= m_end)]
        if idx_m_b.size < 1:
            dropped[str(m)] = "no_bdays_in_month"
            continue
        y_true = float(S_b.loc[idx_m_b].mean())

        # Horizon = full calendar month length (inclusive)
        H = (m_end.date() - m_start.date()).days + 1
        if H <= 0 or H > CFG.max_horizon:
            dropped[str(m)] = f"horizon_invalid(H={H})"
            continue

        # Context truncation
        context = min(CFG.max_context, len(hist_d))
        x = hist_d.values[-context:]

        # TimesFM forecast (point forecast only, ignore quantiles)
        point_forecast, _ = tfm.forecast([x], freq=[0])
        pf = np.asarray(point_forecast[0])

        if pf.shape[0] < H:
            raise RuntimeError(
                f"The model returned a too short horizon (pf={pf.shape[0]}) for H={H}. "
                f"Increase 'horizon_len' in build_model(), e.g. to 64."
            )

        pf = pf[:H]
        f_idx = pd.date_range(cut + pd.Timedelta(days=1), periods=H, freq="D")
        pred_daily = pd.Series(pf, index=f_idx, name="point")

        # Aggregate to business days in the month
        pred_b = pred_daily.reindex(idx_m_b, method=None)
        if pred_b.isna().all():
            dropped[str(m)] = "no_overlap_pred_B_days"
            continue
        y_pred = float(pred_b.dropna().mean())

        rows[str(m)] = {
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
            print("\nDropped months and reasons:")
            for m in miss:
                print(f"  {m}: {dropped.get(m, 'unknown')}")
    return df

# -----------------------------
# Evaluation (level + direction)
# -----------------------------
def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute level errors (RMSE, MAE) and directional accuracy of monthly means.
    """
    df = df.copy()
    df["err"] = df["y_true"] - df["y_pred"]
    eval_df = df.dropna(subset=["y_true", "y_pred"]).copy()

    n_obs = int(len(eval_df))
    rmse = float(np.sqrt(np.mean(np.square(eval_df["err"])))) if n_obs else np.nan
    mae  = float(mean_absolute_error(eval_df["y_true"], eval_df["y_pred"])) if n_obs else np.nan

    eval_df["y_prev"] = eval_df["y_true"].shift(1)
    mask = eval_df["y_prev"].notna()
    dir_true = np.sign(eval_df.loc[mask, "y_true"] - eval_df.loc[mask, "y_prev"])
    dir_pred = np.sign(eval_df.loc[mask, "y_pred"] - eval_df.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    hit_rate = (hits / total) if total else np.nan

    print("\n=== Model performance (monthly mean, EUR/NOK) ===")
    print(f"Observations: {n_obs}")
    print(f"RMSE (level): {rmse:.6f}")
    print(f"MAE  (level): {mae:.6f}")
    if total:
        print(f"Directional accuracy: {hits}/{total} ({hit_rate*100:.1f}%)")

    return eval_df

# -----------------------------
# Diebold–Mariano (vs Random Walk)
# -----------------------------
def _normal_cdf(z: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def dm_test(
    y_true: pd.Series,
    y_model: pd.Series,
    y_rw: pd.Series,
    h: int = 1,
    loss: str = "mse"
) -> Tuple[float, float]:
    """
    Diebold–Mariano test for equal predictive accuracy.
    y_true, y_model, y_rw: aligned and equally long series.
    h: forecast horizon (1 for one-step-ahead monthly).
    loss: "mse" or "mae".
    Returns (DM statistic, p-value).
    """
    df = pd.concat({"y": y_true, "m": y_model, "rw": y_rw}, axis=1).dropna()
    if df.empty or len(df) < 5:
        return float("nan"), float("nan")

    e_m = df["y"] - df["m"]
    e_r = df["y"] - df["rw"]
    if loss.lower() == "mae":
        d = np.abs(e_m) - np.abs(e_r)
    else:  # MSE (squared error)
        d = (e_m ** 2) - (e_r ** 2)

    N = int(len(d))
    d_mean = float(d.mean())

    # HAC variance (Newey–West with Bartlett kernel up to h-1)
    gamma0 = float(np.var(d, ddof=1)) if N > 1 else 0.0
    var_bar = gamma0 / N
    if h > 1 and N > 2:
        for k in range(1, min(h - 1, N - 1) + 1):
            w_k = 1.0 - k / h  # Bartlett weight
            cov_k = float(np.cov(d[k:], d[:-k], ddof=1)[0, 1])
            var_bar += 2.0 * w_k * cov_k / N

    if var_bar <= 0 or not np.isfinite(var_bar):
        return float("nan"), float("nan")

    dm_stat = d_mean / math.sqrt(var_bar)
    # two-sided p-value
    p_val = 2.0 * (1.0 - _normal_cdf(abs(dm_stat)))
    return dm_stat, p_val

def dm_against_random_walk(eval_df: pd.DataFrame, loss: str = "mse", h: int = 1) -> None:
    """
    Random walk benchmark: previous month's observed level (y_{t-1}).
    """
    df = eval_df.copy()
    df["rw_pred"] = df["y_true"].shift(1)
    dm_stat, p_val = dm_test(df["y_true"], df["y_pred"], df["rw_pred"], h=h, loss=loss)
    print("\n=== Diebold–Mariano vs Random Walk ===")
    print(f"Loss: {loss.upper()} | horizon h={h}")
    print(f"DM-statistic: {dm_stat:.4f}" if np.isfinite(dm_stat) else "DM-statistic: nan")
    print(f"p-value     : {p_val:.4f}" if np.isfinite(p_val) else "p-value     : nan")

# -----------------------------
# Plot – simple line plot (no bands)
# -----------------------------
def plot_monthly_simple(eval_df: pd.DataFrame, png_path: str, pdf_path: str):
    """
    Simple line plot of actual vs TimesFM forecasted monthly means.
    """
    if eval_df.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    plt.plot(x, eval_df["y_true"], color="black", label="Actual (monthly mean, B-days)")
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--", label="Forecast (TimesFM)")

    plt.title("TimesFM Forecast vs Actual (Monthly Mean, EUR/NOK)")
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

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Data
    S_b, S_d = load_series(CFG.url)
    if CFG.verbose:
        print(f"Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")
        print(f"Data (D): {S_d.index.min().date()} → {S_d.index.max().date()} | n={len(S_d)}")

    # 2) Model
    tfm = build_model(max_context=CFG.max_context, horizon_len=min(CFG.max_horizon, 64))

    # 3) Walk-forward evaluation (monthly)
    df_eval = walk_forward_timesfm_monthly(S_b, S_d, tfm)
    eval_df = evaluate(df_eval)

    # 4) Diebold–Mariano test vs random walk (MSE; h=1)
    dm_against_random_walk(eval_df, loss="mse", h=1)

    # 5) Plot
    plot_monthly_simple(eval_df, CFG.fig_png, CFG.fig_pdf)

if __name__ == "__main__":
    main()
