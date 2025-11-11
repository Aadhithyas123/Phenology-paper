# src/prep_data.py
import argparse
import os
from pathlib import Path
import sys
import logging
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# sys.path so we can import from src/*
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("eurocropsml.prep")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------
def _deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(paths):
    import yaml
    cfg = {}
    for p in paths:
        with open(p, "r") as f:
            part = yaml.safe_load(f)
        cfg = _deep_merge(cfg, part)
    return cfg

# ---------------------------------------------------------------------
# Import loader helpers from your dataloader
# ---------------------------------------------------------------------
from dataloader import (
    read_split_to_df,
    _fix_path,
    _load_npz_full,    # returns (ts [T,C], dates or None)
)

# ---------------------------------------------------------------------
# Index computations (assume S2 order B1..B12 with/without B10)
# Red=B4 idx=3, NIR=B8 idx=7, B5 idx=4, B8A idx=8, B11 idx=10
# ---------------------------------------------------------------------
def _safe_div(n, d):
    d = np.where(np.abs(d) < 1e-8, 1.0, d)
    return n / d

def _smooth_moving_avg(x, win=7):
    if x is None or len(x) == 0 or win <= 1:
        return x
    w = int(max(1, win))
    w = w + 1 if (w % 2 == 0) else w  # odd window
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(w, dtype=float) / float(w)
    y = np.convolve(xp, k, mode="valid")
    return y.astype(np.float32)

def ndvi_01(ts: np.ndarray) -> np.ndarray:
    red = ts[:, 3]
    nir = ts[:, 7]
    ndvi = _safe_div(nir - red, nir + red)  # [-1,1]
    ndvi01 = 0.5 * (ndvi + 1.0)            # [0,1]
    return np.clip(ndvi01, 0.0, 1.0)

def evi2_01(ts: np.ndarray) -> np.ndarray:
    # EVI2 = 2.5*(NIR-Red)/(NIR + 2.4*Red + 1)
    red = ts[:, 3]
    nir = ts[:, 7]
    num = 2.5 * (nir - red)
    den = nir + 2.4 * red + 1.0
    evi2 = _safe_div(num, den)          # ~[-1,1] in practice
    evi2 = (evi2 - evi2.min()) / max(1e-8, (evi2.max() - evi2.min()))
    return np.clip(evi2, 0.0, 1.0)

def ndre_01(ts: np.ndarray) -> np.ndarray:
    # (B8A - B5) / (B8A + B5)
    b5  = ts[:, 4]
    b8a = ts[:, 8]
    ndre = _safe_div(b8a - b5, b8a + b5)  # [-1,1]
    ndre01 = 0.5 * (ndre + 1.0)
    return np.clip(ndre01, 0.0, 1.0)

def ndwi_01(ts: np.ndarray) -> np.ndarray:
    # (B8 - B11) / (B8 + B11)
    b8  = ts[:, 7]
    b11 = ts[:, 10]
    ndwi = _safe_div(b8 - b11, b8 + b11)  # [-1,1] moisture proxy
    ndwi01 = 0.5 * (ndwi + 1.0)
    return np.clip(ndwi01, 0.0, 1.0)

def msi_01(ts: np.ndarray) -> np.ndarray:
    # MSI = B11 / B8; remap to [0,1] by min-max per parcel
    b8  = ts[:, 7]
    b11 = ts[:, 10]
    msi = _safe_div(b11, np.where(np.abs(b8) < 1e-8, 1.0, b8))
    # normalize per-series
    m0, m1 = float(np.nanmin(msi)), float(np.nanmax(msi))
    rng = max(1e-8, m1 - m0)
    msi01 = (msi - m0) / rng
    return np.clip(msi01, 0.0, 1.0)

# ---------------------------------------------------------------------
# Events on a 0..1 bounded signal
# SOS/EOS via threshold f * max, Peak via argmax
# ---------------------------------------------------------------------
def estimate_events_from_signal(sig01: np.ndarray, frac_of_max: float = 0.1):
    if sig01 is None or len(sig01) == 0 or not np.isfinite(sig01).any():
        return {"SOS": None, "Peak": None, "EOS": None}
    s = np.nan_to_num(sig01, nan=float(np.nanmin(sig01)))
    smax = float(np.max(s))
    if smax <= 1e-8:
        return {"SOS": None, "Peak": int(np.argmax(s)), "EOS": None}
    thr = frac_of_max * smax
    above = np.where(s >= thr)[0]
    SOS = int(above[0]) if above.size else None
    EOS = int(above[-1]) if above.size else None
    Peak = int(np.argmax(s))
    return {"SOS": SOS, "Peak": Peak, "EOS": EOS}

# quick pulse count for moisture/irrigation proxies
def count_pulses(sig01: np.ndarray, min_prominence: float = 0.05) -> int:
    if sig01 is None or len(sig01) < 3:
        return 0
    s = sig01.astype(np.float32)
    # simple prominence heuristic: local maxima higher than neighbors by min_prominence
    peaks = 0
    for i in range(1, len(s) - 1):
        if s[i] > s[i-1] and s[i] > s[i+1]:
            left = s[i] - s[i-1]
            right = s[i] - s[i+1]
            if left >= min_prominence and right >= min_prominence:
                peaks += 1
    return int(peaks)

def lead_lag(a01: np.ndarray, b01: np.ndarray, max_lag: int = 30):
    """
    Return lag* (in indices) maximizing time-lagged correlation of (a vs b).
    Positive lag means 'b lags a' (b peaks after a).
    """
    if a01 is None or b01 is None:
        return None
    a = a01 - np.nanmean(a01)
    b = b01 - np.nanmean(b01)
    a = np.nan_to_num(a, nan=0.0); b = np.nan_to_num(b, nan=0.0)
    best_lag, best_corr = 0, -1.0
    L = int(max_lag)
    for lag in range(-L, L + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:len(b)+lag]
        elif lag > 0:
            aa = a[:len(a)-lag]
            bb = b[lag:]
        else:
            aa, bb = a, b
        if len(aa) < 3 or len(bb) < 3:
            continue
        num = float(np.dot(aa, bb))
        den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        corr = num / den if den > 1e-8 else 0.0
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return int(best_lag)

# ---------------------------------------------------------------------
# Feature summarization
# ---------------------------------------------------------------------
def summarize_timeseries(ts: np.ndarray, pid: str, label: str, country: str, path: str):
    T, C = ts.shape
    means = np.nanmean(ts, axis=0)
    stds  = np.nanstd(ts, axis=0) + 1e-8
    last  = ts[-1, :]
    row = {
        "pid": pid, "label": str(label), "country": str(country),
        "path": path, "T": int(T), "C": int(C),
    }
    for c in range(C):
        row[f"mean_c{c}"] = float(means[c])
        row[f"std_c{c}"]  = float(stds[c])
        row[f"last_c{c}"] = float(last[c])
    return row

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True,
                    help="YAML configs: default first, then overrides (e.g., LV→EE).")
    ap.add_argument("--drop-b10", action="store_true",
                    help="Drop B10 when *modeling*. Here we keep raw for checks, "
                         "but we will validate index band positions accordingly.")
    ap.add_argument("--compute-indices", action="store_true",
                    help="Compute NDVI/NDRE/EVI2/NDWI/MSI series and event metrics.")
    ap.add_argument("--smooth-win", type=int, default=9,
                    help="Odd window length for moving average smoothing (default 9).")
    args = ap.parse_args()

    cfg = load_config(args.config)

    root_npz    = cfg["dataset"]["root_npz"]
    master_json = cfg["dataset"]["master_json"]

    # Countries used in this experiment (pretrain + test)
    all_countries = (
        cfg["dataset"]["countries"]["train"]
        + cfg["dataset"]["countries"].get("extra_train", [])
        + cfg["dataset"]["countries"]["test"]
    )
    keep_countries = {c.upper() for c in all_countries}

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_feats  = processed_dir / "parcel_features.parquet"
    out_events = processed_dir / "events.parquet"

    # Gather all split entries (train/val/test combined)
    df_all = read_split_to_df(
        split_file=master_json,
        split_subset=None,
        countries=None,
        only_overlap_with=None,
        unknown_label_list=None,
    )
    # Resolve absolute paths
    df_all["path"] = df_all["fname"].map(lambda s: _fix_path(root_npz, s))
    # Country filter
    df_all = df_all[df_all["country"].str.upper().isin(keep_countries)].reset_index(drop=True)

    logger.info("Preparing features/events for %d files (countries: %s)",
                len(df_all), sorted(keep_countries))

    feature_rows = []
    event_rows   = []
    n_bad_dates = n_bad_shape = n_bad_ndvi = 0

    for i, row in df_all.iterrows():
        ts_path = row["path"]
        pid  = os.path.splitext(os.path.basename(ts_path))[0]
        label= row["label"]
        ctry = row["country"]

        try:
            # Load raw (no drop) to keep band indices consistent for checks
            ts, dates = _load_npz_full(ts_path, expect_13_or_12_channels=True, drop_b10=False)
        except Exception as e:
            logger.warning("[prep] skip %s (load error): %s", ts_path, e)
            continue

        T, C = ts.shape
        # ---- timestamps sanity
        has_dates = bool(dates is not None and len(dates) == T and T > 0)
        if not has_dates:
            n_bad_dates += 1
            logger.warning("[prep] %s: missing/empty/mismatched dates (len=%s vs T=%d)",
                           os.path.basename(ts_path), (None if dates is None else len(dates)), T)

        # ---- shape sanity
        if T <= 0 or C not in (12, 13):
            n_bad_shape += 1
            logger.warning("[prep] %s: bad shape [T=%d, C=%d]; expected C in {12,13}",
                           os.path.basename(ts_path), T, C)
            continue

        # ---- NDVI sanity (expects 0..1 after mapping)
        try:
            ndvi = ndvi_01(ts)  # [0,1]
            if not np.isfinite(ndvi).any():
                raise ValueError("NDVI all non-finite")
            # a soft check: unclipped distance from [0,1]
            unclipped = 0.5 * (( ( (ts[:,7]-ts[:,3]) / np.where(np.abs(ts[:,7]+ts[:,3])<1e-8,1.0,ts[:,7]+ts[:,3]) ) + 1.0))
            diff = np.nanmean(np.abs(np.clip(unclipped,0,1) - unclipped))
            if diff > 1e-3:
                n_bad_ndvi += 1
                logger.warning("[prep] %s: NDVI deviates from [0,1] by >1e-3; check band order/data.",
                               os.path.basename(ts_path))
        except Exception as e:
            n_bad_ndvi += 1
            logger.warning("[prep] %s: NDVI check failed: %s", os.path.basename(ts_path), e)

        # ---- Feature summary (per-band)
        feature_rows.append(summarize_timeseries(ts, pid, label, ctry, ts_path))

        # ---- Events (NDVI primary; optionally extras)
        ev = {"pid": pid, "label": str(label), "country": str(ctry), "has_dates": has_dates}

        # NDVI events (always compute for consistency)
        s_ndvi = _smooth_moving_avg(ndvi_01(ts), args.smooth_win)
        ndvi_ev = estimate_events_from_signal(s_ndvi, frac_of_max=0.1)
        ev.update({
            "SOS_ndvi": ndvi_ev["SOS"],
            "Peak_ndvi": ndvi_ev["Peak"],
            "EOS_ndvi": ndvi_ev["EOS"],
        })

        if args.compute_indices:
            # NDRE / EVI2 / NDWI / MSI
            try:
                s_ndre  = _smooth_moving_avg(ndre_01(ts), args.smooth_win)
                s_evi2  = _smooth_moving_avg(evi2_01(ts), args.smooth_win)
                s_ndwi  = _smooth_moving_avg(ndwi_01(ts), args.smooth_win)
                s_msi   = _smooth_moving_avg(msi_01(ts), args.smooth_win)

                ndre_ev = estimate_events_from_signal(s_ndre, frac_of_max=0.1)
                evi2_ev = estimate_events_from_signal(s_evi2, frac_of_max=0.1)

                ev.update({
                    "SOS_ndre": ndre_ev["SOS"], "Peak_ndre": ndre_ev["Peak"], "EOS_ndre": ndre_ev["EOS"],
                    "SOS_evi2": evi2_ev["SOS"], "Peak_evi2": evi2_ev["Peak"], "EOS_evi2": evi2_ev["EOS"],
                    "pulses_ndwi": count_pulses(s_ndwi, min_prominence=0.05),
                    "pulses_msi":  count_pulses(s_msi,  min_prominence=0.05),
                    # lag: positive => NDVI leads (NDWI/MSI follow); negative => NDWI/MSI lead NDVI
                    "lag_ndwi_vs_ndvi": lead_lag(s_ndvi, s_ndwi, max_lag=30),
                    "lag_msi_vs_ndvi":  lead_lag(s_ndvi, s_msi,  max_lag=30),
                })
            except Exception as e:
                logger.warning("[prep] %s: index/event extras failed: %s", os.path.basename(ts_path), e)

        event_rows.append(ev)

        if (i + 1) % 1000 == 0:
            logger.info("Processed %d/%d", i + 1, len(df_all))

    # ---- Write outputs
    pd.DataFrame(feature_rows).to_parquet(out_feats, index=False)
    pd.DataFrame(event_rows).to_parquet(out_events, index=False)

    logger.info("[prep] wrote %d parcel feature rows → %s", len(feature_rows), out_feats)
    logger.info("[prep] wrote %d phenology rows      → %s", len(event_rows), out_events)
    logger.info("[prep] sanity summary: bad_dates=%d, bad_shape=%d, bad_ndvi=%d",
                n_bad_dates, n_bad_shape, n_bad_ndvi)
    logger.info("[prep] done.")

if __name__ == "__main__":
    main()
