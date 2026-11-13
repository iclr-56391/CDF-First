#!/usr/bin/env python3
"""
uci_real_loader.py
==================
Real UCI data loading for the CDF-First real-world experiment.
"""

import io
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
    _HAS_UCIMLREPO = True
except Exception:
    _HAS_UCIMLREPO = False

# UCI repo numeric IDs
UCI_IDS = {
    "concrete":    165,
    "energy":      242,
    "parkinsons":  189,
    "air":         360,
    "temperature": 514,
    # 'skillcraft' handled via direct URL below.
}

# Direct download URLs.
# GitHub mirrors are tried FIRST: archive.ics.uci.edu has enforced
# Cloudflare bot-protection since 2024 that returns HTTP 403 to many
# scripted/non-browser requests (affects urllib, requests, and the
# official ucimlrepo package alike, from various networks). Each mirror
# below was verified to serve the correct schema for its dataset before
# being added. The original archive.ics.uci.edu URLs are kept as a
# fallback in case bot-protection does not trigger on your network.
DIRECT_URLS = {
    "concrete": [
        "https://raw.githubusercontent.com/ronneyjes93/concretedataset/main/Concrete_Data.csv",
        "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
    ],
    "energy": [
        "https://raw.githubusercontent.com/rrbrana10/green-tech-ai-assistant/main/ENB2012_data.csv",
        "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
    ],
    "parkinsons": [
        "https://raw.githubusercontent.com/shrishashegde/Parkinsons-Disease-Telemonitoring/main/parkinsons_updrs.csv",
        "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
    ],
    "air": [
        "https://raw.githubusercontent.com/Gauhar1107/AirQualityUCI/master/AirQualityUCI.csv",
        "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
    ],
    "temperature": [
        "https://raw.githubusercontent.com/Sairaghav1999/Dataset/main/Bias_correction_ucl.csv",
        "https://archive.ics.uci.edu/static/public/514/bias+correction+of+numerical+prediction+model+temperature+forecast.zip",
    ],
    "skillcraft": [
        "https://raw.githubusercontent.com/AleEnet/Skillcraft-Dataset/main/SkillCraft1_Dataset.csv",
        "https://archive.ics.uci.edu/static/public/272/skillcraft1+master+table+dataset.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv",
    ],
    "fish": [
        # Fish Market dataset (Weight regression). Verified working mirrors
        # (schema confirmed: Species,Weight,Length1,Length2,Length3,Height,Width; 159 rows).
        "https://raw.githubusercontent.com/lokeshbhirud1502/Fish-Weight-prediction-/master/Fish.csv",
        "https://raw.githubusercontent.com/naman-dhammi/Fish-Species-Weight-Prediction/main/Fish.csv",
    ],
}

_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "*/*",
}

def _download_one(url: str, cache_dir: str) -> bytes:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # unique cache name per URL to avoid collisions between mirrors
    safe = url.split("?")[0].rstrip("/").split("/")[-1] or "download.dat"
    fname = Path(cache_dir) / safe
    if fname.exists() and fname.stat().st_size > 0:
        return fname.read_bytes()
    print(f"    downloading {url} ...")
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    # reject obvious HTML error pages masquerading as data
    head = data[:200].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("server returned an HTML page, not a data file")
    fname.write_bytes(data)
    return data

def _download(urls, cache_dir: str) -> bytes:
    if isinstance(urls, str):
        urls = [urls]
    last = None
    for u in urls:
        try:
            return _download_one(u, cache_dir)
        except Exception as e:
            last = e
            print(f"      (failed: {type(e).__name__}: {e})")
    raise RuntimeError(f"All download URLs failed. Last error: {last}")

def _bytes_from_zip(raw: bytes, want_ext=(".csv", ".xlsx", ".xls", ".data", ".txt")):
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(want_ext) and not n.startswith("__")]
            if not names:
                names = [n for n in z.namelist() if not n.endswith("/")]
            with z.open(names[0]) as f:
                return names[0], f.read()
    return None, raw

def _finalize(X: np.ndarray, y: np.ndarray,
              d_x: int, d_y: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    good = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    X, y = X[good], y[good]
    assert X.shape[1] == d_x, f"[{name}] expected d_x={d_x}, got {X.shape[1]}"
    assert y.shape[1] == d_y, f"[{name}] expected d_y={d_y}, got {y.shape[1]}"
    print(f"    [{name}] loaded real data: X={X.shape}, y={y.shape}")
    return X.astype(np.float32), y.astype(np.float32)

def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every column to float, encoding any non-numeric column
    (object, pandas StringDtype, category, etc.) as integer codes.
    Uses pd.api.types.is_numeric_dtype rather than an explicit dtype
    check, since modern pandas may return StringDtype (not plain
    'object') for text columns, which a strict `== object` check misses.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = pd.Categorical(out[c]).codes.astype(float)
    return out

def _read_table(raw: bytes, **kw) -> pd.DataFrame:
    name, data = _bytes_from_zip(raw)
    ext = (name or "").lower()
    bio = io.BytesIO(data)
    if ext.endswith((".xlsx",)):
        return pd.read_excel(bio, engine="openpyxl")
    if ext.endswith((".xls",)):
        return pd.read_excel(bio)
    if data[:2] == b"PK" or data[:4] == b"\xd0\xcf\x11\xe0":
        try:
            return pd.read_excel(io.BytesIO(data))
        except Exception:
            pass
    text = data.decode("latin-1", errors="ignore")
    sep = ";" if text.count(";") > text.count(",") else ","
    default = dict(sep=sep)
    default.update(kw)
    return pd.read_csv(io.BytesIO(data), **default)

def _load_fish(cache_dir, d_x, d_y):
    """
    Fish Market dataset. Columns:
        Species, Weight, Length1, Length2, Length3, Height, Width
    Target = Weight (d_y=1). Features = the other 6 (Species encoded) (d_x=6).
    """
    raw = _download(DIRECT_URLS["fish"], cache_dir)
    df = _read_table(raw)
    # identify target BEFORE numeric encoding (case-insensitive)
    weight_col = next((c for c in df.columns if str(c).strip().lower() == "weight"), None)
    if weight_col is None:
        raise RuntimeError(f"'Weight' column not found in fish data; columns={list(df.columns)}")
    df = _numeric_only(df).dropna()          # Species -> integer codes
    feat = [c for c in df.columns if c != weight_col][:d_x]
    X = df[feat].values
    y = df[[weight_col]].values
    return _finalize(X, y, d_x, d_y, "fish")

def _load_concrete(cache_dir, d_x, d_y):
    raw = _download(DIRECT_URLS["concrete"], cache_dir)
    df = _read_table(raw)
    df = _numeric_only(df).dropna()
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8:9].values
    return _finalize(X, y, d_x, d_y, "concrete")

def _load_energy(cache_dir, d_x, d_y):
    raw = _download(DIRECT_URLS["energy"], cache_dir)
    df = _read_table(raw)
    df = _numeric_only(df).dropna()
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8:10].values
    return _finalize(X, y, d_x, d_y, "energy")

def _load_parkinsons(cache_dir, d_x, d_y):
    raw = _download(DIRECT_URLS["parkinsons"], cache_dir)
    df = _read_table(raw)
    df = _numeric_only(df).dropna()
    target_cols = ["motor_UPDRS", "total_UPDRS"]
    drop_for_x = ["subject#"] + target_cols
    feat = [c for c in df.columns if c not in drop_for_x][:d_x]
    X = df[feat].values
    y = df[target_cols].values
    return _finalize(X, y, d_x, d_y, "parkinsons")

def _load_air(cache_dir, d_x, d_y):
    raw = _download(DIRECT_URLS["air"], cache_dir)
    df = _read_table(raw, decimal=",")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    for c in ["Date", "Time"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = _numeric_only(df).replace(-200, np.nan).dropna()
    target_cols = [c for c in ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)"]
                   if c in df.columns][:d_y]
    if len(target_cols) < d_y:
        target_cols = list(df.columns[:d_y])
    feat = [c for c in df.columns if c not in target_cols][:d_x]
    X = df[feat].values
    y = df[target_cols].values
    return _finalize(X, y, d_x, d_y, "air")

def _load_temperature(cache_dir, d_x, d_y):
    raw = _download(DIRECT_URLS["temperature"], cache_dir)
    df = _read_table(raw)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = _numeric_only(df).dropna()
    target_cols = [c for c in df.columns
                   if "Next_Tmax" in str(c) or "Next_Tmin" in str(c)][:d_y]
    if len(target_cols) < d_y:
        target_cols = list(df.columns[-d_y:])
    feat = [c for c in df.columns if c not in target_cols][:d_x]
    X = df[feat].values
    y = df[target_cols].values
    return _finalize(X, y, d_x, d_y, "temperature")

def _load_skillcraft(cache_dir, d_x, d_y):
    raw = _download(DIRECT_URLS["skillcraft"], cache_dir)
    df = _read_table(raw)
    df = _numeric_only(df).dropna()
    target_cols = [c for c in ["APM", "ActionLatency",
                               "GapBetweenPACs", "TotalMapExplored"]
                   if c in df.columns]
    if len(target_cols) < d_y:
        target_cols = list(df.columns[-d_y:])
    target_cols = target_cols[:d_y]
    feat = [c for c in df.columns if c not in target_cols]
    if "GameID" in feat:
        feat.remove("GameID")
    feat = feat[:d_x]
    X = df[feat].values
    y = df[target_cols].values
    return _finalize(X, y, d_x, d_y, "skillcraft")

_LOADERS = {
    "concrete":    _load_concrete,
    "energy":      _load_energy,
    "parkinsons":  _load_parkinsons,
    "air":         _load_air,
    "temperature": _load_temperature,
    "skillcraft":  _load_skillcraft,
    "fish":        _load_fish,
}

def load_real_uci(dataset_name: str, d_x: int, d_y: int,
                  cache_dir: str = "./uci_cache") -> Tuple[np.ndarray, np.ndarray]:
    name = dataset_name.lower()
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Available: {sorted(_LOADERS)}")

    if _HAS_UCIMLREPO and name in UCI_IDS:
        try:
            ds = fetch_ucirepo(id=UCI_IDS[name])
            Xdf = _numeric_only(ds.data.features)
            ydf = _numeric_only(ds.data.targets)
            df = pd.concat([Xdf, ydf], axis=1).dropna()
            Xdf = df[Xdf.columns]
            ydf = df[ydf.columns]
            Xv = Xdf.values
            yv = ydf.values
            if yv.shape[1] > d_y:
                yv = yv[:, :d_y]
            if Xv.shape[1] > d_x:
                Xv = Xv[:, :d_x]
            if Xv.shape[1] == d_x and yv.shape[1] == d_y:
                return _finalize(Xv, yv, d_x, d_y, name + " (ucimlrepo)")
            print(f"    [{name}] ucimlrepo column mismatch "
                  f"(X={Xv.shape[1]}, y={yv.shape[1]}); using direct loader")
        except Exception as e:
            print(f"    [{name}] ucimlrepo failed ({e}); using direct loader")

    return _LOADERS[name](cache_dir, d_x, d_y)


if __name__ == "__main__":
    CONFIGS = {
        "fish": (6, 1), "concrete": (8, 1), "energy": (8, 2),
        "parkinsons": (16, 2), "temperature": (21, 2),
        "air": (10, 3), "skillcraft": (15, 4),
    }
    print("Testing real UCI loaders (ucimlrepo available: %s)\n" % _HAS_UCIMLREPO)
    for name, (dx, dy) in CONFIGS.items():
        try:
            X, y = load_real_uci(name, dx, dy)
            print(f"  OK  {name:<12} X={X.shape} y={y.shape}  "
                  f"y-range=[{y.min():.2f}, {y.max():.2f}]")
        except Exception as e:
            print(f"  FAIL {name:<12} {e}")
