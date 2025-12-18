# -*- coding: utf-8 -*-
"""
TimesFM 2.0 (500M, PyTorch) – EUR/NOK walk-forward (quarterly).
Source: Norges Bank CSV (semicolon separated, decimal comma)
No intervals (point forecast only).

This module contains both the shared quarterly evaluation helpers and
an executable main that runs the EUR/NOK benchmark.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import certifi
import matplotlib.pyplot as plt

# Prefer local TimesFM v1 API (timesfm/v1/src) to avoid conflicts
import sys
_ROOT = Path(__file__).resolve().parent.parent
_V1_SRC = _ROOT / "timesfm" / "v1" / "src"
if _V1_SRC.exists():
    sys.path.insert(0, str(_V1_SRC))
import timesfm  # v1 API (TimesFm, TimesFmHparams, TimesFmCheckpoint)


LossType = Literal["mse", "mae"]
ForecastDailyFn = Callable[[np.ndarray, int], np.ndarray]


# -----------------------------
# Config (quarterly)
# -----------------------------
@dataclass(frozen=True)
class EvalConfigQ:
    url: str
    series: str = "EUR_NOK"
    q_freq: str = "Q-DEC"
    min_hist_days: int = 40
    max_context: int = 2048
    max_horizon: int = 512
    retries: int = 3
    timeout: int = 60
    verbose: bool = True
    # CSV dialect for Norges Bank
    date_col: str = "TIME_PERIOD"
    value_col: str = "OBS_VALUE"
    csv_sep: str = ";"
    csv_decimal: str = ","
    csv_encoding: str = "utf-8-sig"


# -----------------------------
# Download + Data (quarterly)
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


def load_series_q(cfg: EvalConfigQ) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      S_b: business-day (B) series with ffill (for cuts and quarterly truth)
      S_d: daily (D) series with ffill (for model context)

    Supports two input dialects:
      1) GitHub daily panel with columns ['Date', cfg.series]
      2) Norges Bank CSV with columns [cfg.date_col, cfg.value_col]
    """
    text = download_csv_text(cfg.url, cfg.retries, cfg.timeout)

    # Attempt GitHub daily panel first
    raw = pd.read_csv(io.StringIO(text))
    if ("Date" in raw.columns) and (cfg.series in raw.columns):
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
    else:
        # Fallback to Norges Bank dialect
        raw_nb = pd.read_csv(
            io.StringIO(text),
            sep=cfg.csv_sep,
            decimal=cfg.csv_decimal,
            encoding=cfg.csv_encoding,
        )
        if not ({cfg.date_col, cfg.value_col} <= set(raw_nb.columns)):
            raise ValueError(
                f"CSV does not contain expected columns for either dialect. Got: {list(raw_nb.columns)}"
            )
        df = (
            raw_nb[[cfg.date_col, cfg.value_col]]
            .rename(columns={cfg.date_col: "DATE", cfg.value_col: cfg.series})
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
# Walk-forward (quarterly) generic runner
# -----------------------------
def walk_forward_q(
    S_b: pd.Series,
    S_d: pd.Series,
    cfg: EvalConfigQ,
    forecast_daily_fn: ForecastDailyFn,
) -> pd.DataFrame:
    """
    Generic quarterly walk-forward:
      - cut = last B-day of prev quarter
      - y_true = mean of S_b in quarter q (business days)
      - y_pred = mean of predicted daily values on business days in quarter q
      - rw_pred = driftless RW level at cut (S_b.loc[cut])
    """
    first_q = pd.Period(S_b.index.min(), freq=cfg.q_freq)
    last_q = pd.Period(S_b.index.max(), freq=cfg.q_freq)
    quarters = pd.period_range(first_q, last_q, freq=cfg.q_freq)

    rows: Dict[str, Dict] = {}
    dropped: Dict[str, str] = {}

    for q in quarters:
        prev_q = q - 1
        q_start, q_end = q.start_time, q.end_time
        prev_start, prev_end = prev_q.start_time, prev_q.end_time

        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(q)] = "no_cut_in_prev_quarter"
            continue

        hist_d = S_d.loc[:cut]
        if hist_d.size < cfg.min_hist_days:
            dropped[str(q)] = f"hist<{cfg.min_hist_days}"
            continue

        idx_q_b = S_b.index[(S_b.index >= q_start) & (S_b.index <= q_end)]
        if idx_q_b.size < 1:
            dropped[str(q)] = "no_bdays_in_quarter"
            continue

        y_true = float(S_b.loc[idx_q_b].mean())
        rw_pred = float(S_b.loc[cut])

        H = (q_end.date() - q_start.date()).days + 1
        if H <= 0 or H > cfg.max_horizon:
            dropped[str(q)] = f"horizon_invalid(H={H})"
            continue

        context = min(cfg.max_context, len(hist_d))
        x = hist_d.values[-context:]

        pf = np.asarray(forecast_daily_fn(x, H), dtype=float)
        if pf.shape[0] < H:
            dropped[str(q)] = f"horizon_short({pf.shape[0]})"
            continue
        pf = pf[:H]

        f_idx = pd.date_range(cut + pd.Timedelta(days=1), periods=H, freq="D")
        pred_daily = pd.Series(pf, index=f_idx, name="point")

        pred_b = pred_daily.reindex(idx_q_b, method=None)
        if pred_b.isna().all():
            dropped[str(q)] = "no_overlap_pred_B_days"
            continue

        y_pred = float(pred_b.dropna().mean())

        rows[str(q)] = {"quarter": q, "cut": cut, "y_true": y_true, "y_pred": y_pred, "rw_pred": rw_pred}

    df = pd.DataFrame.from_dict(rows, orient="index")
    if not df.empty:
        df = df.set_index("quarter").sort_index()

    if cfg.verbose and dropped:
        miss = [str(q) for q in quarters if q not in df.index]
        if miss:
            print("\nDropped quarters and reasons:")
            for qq in miss:
                print(f"  {qq}: {dropped.get(qq, 'unknown')}")
    return df


# -----------------------------
# Metrics + DM (quarterly)
# -----------------------------
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


def dm_against_rw_q(eval_df: pd.DataFrame, loss: LossType = "mse", h: int = 1) -> None:
    if "rw_pred" not in eval_df.columns:
        raise ValueError("Missing 'rw_pred'. Ensure walk_forward_q() output is used.")
    dm_stat, p_val = dm_test(eval_df["y_true"], eval_df["y_pred"], eval_df["rw_pred"], h=h, loss=loss)
    print("\n=== Diebold–Mariano vs Random Walk (driftless, cut-level) ===")
    print(f"Loss: {loss.upper()} | horizon h={h}")
    print(f"DM-statistic: {dm_stat:.4f}" if np.isfinite(dm_stat) else "DM-statistic: nan")
    print(f"p-value     : {p_val:.4f}" if np.isfinite(p_val) else "p-value     : nan")


# -----------------------------
# Plot (quarterly)
# -----------------------------
def evaluate_q(eval_df: pd.DataFrame) -> pd.DataFrame:
    df = eval_df.copy()
    df["err"] = df["y_true"] - df["y_pred"]
    core = df.dropna(subset=["y_true", "y_pred"]).copy()

    n_obs = int(len(core))
    rmse = float(np.sqrt(np.mean(np.square(core["err"])))) if n_obs else np.nan
    mae = float(np.mean(np.abs(core["err"]))) if n_obs else np.nan

    core["y_prev"] = core["y_true"].shift(1)
    mask = core["y_prev"].notna()
    dir_true = np.sign(core.loc[mask, "y_true"] - core.loc[mask, "y_prev"])
    dir_pred = np.sign(core.loc[mask, "y_pred"] - core.loc[mask, "y_prev"])
    hits = int((dir_true.values == dir_pred.values).sum())
    total = int(mask.sum())
    hit_rate = (hits / total) if total else np.nan

    print("\n=== Model performance (quarterly mean, EUR/NOK) ===")
    print(f"Observations: {n_obs}")
    print(f"RMSE (level): {rmse:.6f}")
    print(f"MAE  (level): {mae:.6f}")
    if total:
        print(f"Directional accuracy: {hits}/{total} ({hit_rate*100:.1f}%)")

    return core


# -----------------------------
# Plot (quarterly)
# -----------------------------
def plot_q(
    eval_df: pd.DataFrame,
    title: str,
    png_path: str,
    pdf_path: str,
    forecast_label: str = "Forecast",
) -> None:
    if eval_df.empty:
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    plt.plot(x, eval_df["y_true"], color="black", label="Actual (quarterly mean, B-days)")
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--", label=forecast_label)

    plt.title(title)
    plt.xlabel("Quarter")
    plt.ylabel("Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# -----------------------------
# Config (eval_common_quarterly)
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
    value_col="OBS_VALUE",
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
            horizon_len=horizon_len,   # must cover longest quarter (~92 days)
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
    Adapter expected by eval_common_quarterly:
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

    # 3) Walk-forward + evaluation (shared)
    df_eval = walk_forward_q(S_b, S_d, CFG, forecast_daily_fn)
    eval_df = evaluate_q(df_eval)

    # 4) DM vs driftless RW (shared)
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

# Compatibility aliases for scripts expecting older names
EvalConfig = EvalConfigQ
load_series = load_series_q
walk_forward_quarterly = walk_forward_q
evaluate_quarterly = evaluate_q
dm_against_rw = dm_against_rw_q
plot_quarterly = plot_q
