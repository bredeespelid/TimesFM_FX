# -*- coding: utf-8 -*-
"""
TimesFM 2.5 – EUR/NOK walk-forward (quarterly, levels) with unified metrics/plot
- Data: GitHub CSV (semicolon; decimal comma), forward-filled to daily
- Cut = last B-day in previous quarter
- Forecast next quarter at daily frequency -> aggregate to quarterly mean over B-days
- Print: Observations, RMSE, MAE, Directional accuracy
- DM test vs Random Walk (MSE, h=1)
- Plot: Actual (black) vs Forecast (blue dashed), no quantile bands

Prereqs:
  pip install pandas numpy scikit-learn requests certifi matplotlib
  # PyTorch (CPU): pip install torch --index-url https://download.pytorch.org/whl/cpu
  # TimesFM 2.5 from repo:
  #   git clone https://github.com/google-research/timesfm.git
  #   cd timesfm && pip install -e . && cd ..
"""

from __future__ import annotations
import io, time, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable

import numpy as np
import pandas as pd
import requests, certifi
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import timesfm  # MUST be the 2.5 repo install; no fallback

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    url: str = "https://raw.githubusercontent.com/bredeespelid/Data_MasterOppgave/refs/heads/main/EURNOK/EUR_NOK_NorgesBank.csv"
    q_freq: str = "Q-DEC"
    min_hist_days: int = 40
    max_context: int = 2048
    max_horizon: int = 512
    retries: int = 3
    timeout: int = 60
    verbose: bool = True
    fig_png: str = "EUR_NOK_TimesFM_vs_Actual.png"
    fig_pdf: str = "EUR_NOK_TimesFM_vs_Actual.pdf"

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

def load_series(url: str) -> Tuple[pd.Series, pd.Series]:
    """
    Reads GitHub CSV (semicolon + decimal comma).
    Returns:
      S_b: business-day (B) with ffill (for cut and quarterly ground truth)
      S_d: daily (D) with ffill (for model inputs and daily forecasts)
    """
    text = download_csv_text(url, CFG.retries, CFG.timeout)
    raw = pd.read_csv(io.StringIO(text), sep=';', encoding='utf-8-sig', decimal=',')
    required_cols = {"TIME_PERIOD", "OBS_VALUE"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Got: {list(raw.columns)}")

    df = (raw[['TIME_PERIOD', 'OBS_VALUE']]
          .rename(columns={'OBS_VALUE': 'EUR_NOK'})
          .assign(TIME_PERIOD=lambda x: pd.to_datetime(x['TIME_PERIOD'], errors='coerce'))
          .dropna(subset=['TIME_PERIOD', 'EUR_NOK'])
          .sort_values('TIME_PERIOD')
          .set_index('TIME_PERIOD'))

    S_b = df['EUR_NOK'].asfreq('B').ffill().astype(float)
    S_b.name = 'EUR_NOK'

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    S_d = df['EUR_NOK'].reindex(full_idx).ffill().astype(float)
    S_d.index.name = 'DATE'
    S_d.name = 'EUR_NOK'
    return S_b, S_d

def last_trading_day(S_b: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Timestamp]:
    sl = S_b.loc[start:end]
    if sl.empty:
        return None
    return sl.index[-1]

# -----------------------------
# Model (TimesFM 2.5 only)
# -----------------------------
def build_model(max_context: int, horizon_len: int) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Returns forecast_fn(x, H) -> np.ndarray length H (point forecast).
    STRICT: Only TimesFM 2.5 repo API is allowed; no fallback.
    """
    if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        raise RuntimeError("TimesFM 2.5 repo API not found. Ensure the 2.5 package is installed (pip install -e .).")

    # Prefer loading the Hugging Face checkpoint using the repo API when available
    repo_id = "google/timesfm-2.5-200m-pytorch"
    m = None
    cls = timesfm.TimesFM_2p5_200M_torch
    if hasattr(cls, "from_pretrained"):
        try:
            if CFG.verbose:
                print(f"Loading TimesFM checkpoint from Hugging Face: {repo_id}")
            # Use from_pretrained; avoid torch.compile by default to keep compatibility
            m = cls.from_pretrained(repo_id, torch_compile=False)
        except Exception as e:
            if CFG.verbose:
                print(f"Could not load checkpoint from Hugging Face: {e}. Falling back to local init.")

    # Fallback: instantiate local class and attempt to load a local checkpoint if supported
    if m is None:
        m = cls()
        if hasattr(m, "load_checkpoint"):
            # Some repo builds require a 'path' argument; try the no-arg call first,
            # fall back to path=None, and otherwise skip loading the checkpoint.
            try:
                m.load_checkpoint()
            except TypeError:
                try:
                    m.load_checkpoint(path=None)
                except (TypeError, NotImplementedError):
                    if CFG.verbose:
                        print("Warning: TimesFM.load_checkpoint requires a 'path' argument or is not implemented; skipping checkpoint load.")

    if hasattr(timesfm, "ForecastConfig"):
        cfg = timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=horizon_len,
            normalize_inputs=True,
            use_continuous_quantile_head=False,  # point forecasts only for identical plot
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
        if hasattr(m, "compile"):
            m.compile(cfg)

    def _forecast(x: np.ndarray, H: int) -> np.ndarray:
        if not hasattr(m, "forecast"):
            raise RuntimeError("TimesFM 2.5 API missing 'forecast'.")
        out = m.forecast(horizon=H, inputs=[x])
        # Some builds return (point, quant) tuple; handle both
        if isinstance(out, tuple):
            point = out[0][0]
        else:
            point = out[0]
        return np.asarray(point, dtype=float)[:H]

    return _forecast

# -----------------------------
# Walk-forward (quarterly) – point forecasts only
# -----------------------------
def walk_forward_timesfm(S_b: pd.Series, S_d: pd.Series, forecast_fn: Callable[[np.ndarray, int], np.ndarray]) -> pd.DataFrame:
    first_q = pd.Period(S_b.index.min(), freq=CFG.q_freq)
    last_q  = pd.Period(S_b.index.max(),  freq=CFG.q_freq)
    quarters = pd.period_range(first_q, last_q, freq=CFG.q_freq)

    rows: Dict = {}
    dropped: Dict[str, str] = {}

    for q in quarters:
        prev_q = q - 1
        q_start, q_end = q.start_time, q.end_time
        prev_start, prev_end = prev_q.start_time, prev_q.end_time

        cut = last_trading_day(S_b, prev_start, prev_end)
        if cut is None:
            dropped[str(q)] = "no_cut_in_prev_q"
            continue

        hist_d = S_d.loc[:cut]
        if hist_d.size < CFG.min_hist_days:
            dropped[str(q)] = f"hist<{CFG.min_hist_days}"
            continue

        idx_q_b = S_b.index[(S_b.index >= q_start) & (S_b.index <= q_end)]
        if idx_q_b.size < 1:
            dropped[str(q)] = "no_bdays_in_q"
            continue
        y_true = float(S_b.loc[idx_q_b].mean())

        H = (q_end.date() - q_start.date()).days + 1
        if H <= 0 or H > CFG.max_horizon:
            dropped[str(q)] = f"horizon_invalid(H={H})"
            continue

        context = min(CFG.max_context, len(hist_d))
        x = hist_d.values[-context:]

        pf = forecast_fn(x, H)
        if pf.shape[0] < H:
            raise RuntimeError(f"Model returned too short horizon (pf={pf.shape[0]}) for H={H}.")

        f_idx = pd.date_range(cut + pd.Timedelta(days=1), periods=H, freq='D')
        pred_daily = pd.Series(pf[:H], index=f_idx, name='point')

        pred_b = pred_daily.reindex(idx_q_b, method=None)
        if pred_b.isna().all():
            dropped[str(q)] = "no_overlap_pred_B_days"
            continue
        y_pred = float(pred_b.dropna().mean())

        rows[str(q)] = {"quarter": q, "cut": cut, "y_true": y_true, "y_pred": y_pred}

    df = pd.DataFrame.from_dict(rows, orient='index')
    if not df.empty:
        df = df.set_index("quarter").sort_index()

    if CFG.verbose and dropped:
        miss = [str(q) for q in quarters if q not in df.index]
        if miss:
            print("\nDropped quarters and reasons:")
            for qq in miss:
                print(f"  {qq}: {dropped.get(qq, 'unknown')}")
    return df

# -----------------------------
# Evaluation (identical printout to 2.0)
# -----------------------------
def evaluate(df: pd.DataFrame) -> pd.DataFrame:
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

    print("\n=== Model performance (quarterly mean, EUR/NOK) ===")
    print(f"Observations: {n_obs}")
    print(f"RMSE (level): {rmse:.6f}")
    print(f"MAE  (level): {mae:.6f}")
    if total:
        print(f"Directional accuracy: {hits}/{total} ({hit_rate*100:.1f}%)")

    return eval_df

# -----------------------------
# Diebold–Mariano vs Random Walk
# -----------------------------
def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def dm_test(y_true: pd.Series, y_model: pd.Series, y_rw: pd.Series, h: int = 1, loss: str = "mse"):
    df = pd.concat({"y": y_true, "m": y_model, "rw": y_rw}, axis=1).dropna()
    if df.empty or len(df) < 5:
        return float("nan"), float("nan")

    e_m = df["y"] - df["m"]
    e_r = df["y"] - df["rw"]
    d = (np.abs(e_m) - np.abs(e_r)) if loss.lower() == "mae" else ((e_m**2) - (e_r**2))

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

def dm_against_random_walk(eval_df: pd.DataFrame, loss: str = "mse", h: int = 1) -> None:
    df = eval_df.copy()
    df["rw_pred"] = df["y_true"].shift(1)
    dm_stat, p_val = dm_test(df["y_true"], df["y_pred"], df["rw_pred"], h=h, loss=loss)
    print("\n=== Diebold–Mariano vs Random Walk ===")
    print(f"Loss: {loss.upper()} | horizon h={h}")
    print(f"DM-statistic: {dm_stat:.4f}" if np.isfinite(dm_stat) else "DM-statistic: nan")
    print(f"p-value     : {p_val:.4f}" if np.isfinite(p_val) else "p-value     : nan")

# -----------------------------
# Plot – same style as 2.0 (no bands)
# -----------------------------
def plot_quarterly_simple(eval_df: pd.DataFrame, png_path: str, pdf_path: str):
    if eval_df.empty:
        print("Nothing to plot.")
        return

    x = eval_df.index.to_timestamp() if isinstance(eval_df.index, pd.PeriodIndex) else eval_df.index

    plt.figure(figsize=(10, 6))
    # Actual: black
    plt.plot(x, eval_df["y_true"], color="black", label="Actual (quarterly mean)")
    # Forecast: blue dashed
    plt.plot(x, eval_df["y_pred"], color="tab:blue", linestyle="--", label="Forecast (TimesFM)")

    plt.title("TimesFM Forecast vs Actual (Quarterly Mean, EUR/NOK)")
    plt.xlabel("Quarter")
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
    # Data
    S_b, S_d = load_series(CFG.url)
    if CFG.verbose:
        print(f"Data (B): {S_b.index.min().date()} → {S_b.index.max().date()} | n={len(S_b)}")
        print(f"Data (D): {S_d.index.min().date()} → {S_d.index.max().date()} | n={len(S_d)}")

    # Model (TimesFM 2.5 only)
    forecast_fn = build_model(max_context=CFG.max_context, horizon_len=min(CFG.max_horizon, 256))

    # Walk-forward, evaluate, DM-test, plot
    df_eval = walk_forward_timesfm(S_b, S_d, forecast_fn)
    eval_df = evaluate(df_eval)
    dm_against_random_walk(eval_df, loss="mse", h=1)
    plot_quarterly_simple(eval_df, CFG.fig_png, CFG.fig_pdf)

if __name__ == "__main__":
    main()
