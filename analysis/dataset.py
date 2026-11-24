"""
Window-level dataset for the drone-identification experiments (2026-09-26).

Walks every session manifest under data/fixed_rig/ (enrollment) and
data/fixed_rig_test/ (held-out test), takes each stage's accepted attempt,
and turns each converted 1 kHz recording into overlapping 1 s windows with
three parallel representations:

  feat_comb   — the project's 120-D feature vector (features.py) on the
                highpass + comb-filtered residual (chassis-only hypothesis)
  feat_raw    — same 120-D vector on highpass-only data (motor lines kept)
  psd         — log-PSD, 6 channels x 5-495 Hz in 10 Hz bins (294-D), on
                highpass-only data; a representation that makes no
                assumption about which features matter

plus per-window metadata (drone, kind, session, day, purpose, phase, PWM,
vbat, recording id). Everything is cached to data/analysis_cache/ keyed by
the converted file's name + mtime, so reruns are cheap and a re-converted
file is picked up automatically.

Splitting rule used by every experiment: NEVER split windows of the same
recording/session between train and test — consecutive 1 s windows overlap
50 % and are nearly identical, so a random window split measures memory,
not identity.
"""

import json
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from pipeline.preprocessing import load_session, infer_fs, highpass, harmonic_deconvolution, MOTOR_COLS
from pipeline.features import extract_features, feature_matrix, NOISE_COLS

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "analysis_cache"
WIN_S, HOP_S = 1.0, 0.5
PSD_EDGES = np.arange(5, 500, 10)          # 5..495 Hz, 49 bins/channel


def iter_recordings():
    """Yield one dict per accepted recording across enrollment + test trees."""
    for tree in ("fixed_rig", "fixed_rig_test"):
        for f in sorted(glob.glob(str(ROOT / "data" / tree / "drone*" / "session_*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            for r in d["recordings"]:
                acc = [a for a in r["attempts"]
                       if a["status"] in ("matched", "completed") and a.get("converted_path")]
                if not acc:
                    continue
                a = acc[-1]
                yield dict(drone=d["drone"], kind=d.get("kind", "enroll"),
                           session=d["session"], day=d["created_iso"][:10],
                           purpose=r["purpose"],
                           path=str(ROOT / a["converted_path"].replace("\\", "/")))


def _psd_windows(df: pd.DataFrame, fs: float) -> np.ndarray:
    n_win, n_hop = int(WIN_S * fs), int(HOP_S * fs)
    out = []
    for s in range(0, len(df) - n_win + 1, n_hop):
        feats = []
        for c in NOISE_COLS:
            x = df[c].values[s:s + n_win]
            f, p = welch(x - x.mean(), fs=fs, nperseg=256)
            logp = np.log10(np.maximum(p, 1e-20))
            idx = np.digitize(f, PSD_EDGES)
            feats.append([logp[idx == k].mean() if np.any(idx == k) else np.nan
                          for k in range(1, len(PSD_EDGES))])
        out.append(np.concatenate(feats))
    return np.asarray(out)


def _process(rec: dict) -> pd.DataFrame:
    df = load_session(rec["path"])
    fs = infer_fs(df)
    hp = highpass(df, fs)
    comb = harmonic_deconvolution(hp, fs) if hp[MOTOR_COLS].max().max() > 500 else hp

    f_comb = extract_features(comb, fs, WIN_S, HOP_S)
    f_raw = extract_features(hp, fs, WIN_S, HOP_S)
    psd = _psd_windows(hp, fs)
    n = min(len(f_comb), len(f_raw), len(psd))

    # per-window metadata
    n_win, n_hop = int(WIN_S * fs), int(HOP_S * fs)
    pwm_active = df[MOTOR_COLS].max(axis=1).values
    vb = df["vbat_v"].values if "vbat_v" in df else np.full(len(df), np.nan)
    meta = []
    for i in range(n):
        s = i * n_hop
        meta.append(dict(pwm=float(np.mean(pwm_active[s:s + n_win])),
                         vbat=float(np.nanmean(vb[s:s + n_win])),
                         t0=float(f_comb["window_start_s"].iloc[i]),
                         phase=f_comb["phase"].iloc[i]))
    out = pd.DataFrame(meta)
    for k in ("drone", "kind", "session", "day", "purpose"):
        out[k] = rec[k]
    out["rec_id"] = Path(rec["path"]).stem
    out["session_id"] = f"{rec['drone']}-{rec['kind']}-{rec['session']:02d}"

    fc = feature_matrix(f_comb.iloc[:n]); fr = feature_matrix(f_raw.iloc[:n])
    names = [c for c in f_comb.columns if c not in ("window_start_s", "phase")]
    nb = len(PSD_EDGES) - 1
    psd_names = [f"psd:{NOISE_COLS[j // nb].replace('noise.', '')}:{PSD_EDGES[j % nb]}"
                 for j in range(psd.shape[1])]
    blocks = [out,
              pd.DataFrame(fc, columns=[f"comb:{nm}" for nm in names]),
              pd.DataFrame(fr, columns=[f"raw:{nm}" for nm in names]),
              pd.DataFrame(psd[:n], columns=psd_names)]
    return pd.concat(blocks, axis=1)


def load_dataset(verbose: bool = True) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    recs = list(iter_recordings())
    for i, rec in enumerate(recs):
        p = Path(rec["path"])
        key = hashlib.md5(f"{p.name}:{p.stat().st_mtime_ns}:v1".encode()).hexdigest()[:12]
        cf = CACHE / f"{p.stem}_{key}.pkl"
        if cf.exists():
            parts.append(pd.read_pickle(cf))
            continue
        if verbose:
            print(f"[{i+1}/{len(recs)}] {p.name}", flush=True)
        dfw = _process(rec)
        dfw.to_pickle(cf)
        parts.append(dfw)
    return pd.concat(parts, ignore_index=True)


def cols(df: pd.DataFrame, rep: str) -> "list[str]":
    """Feature columns of one representation: 'comb', 'raw' or 'psd'."""
    return [c for c in df.columns if c.startswith(rep + ":")]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    ds = load_dataset()
    print(ds.shape)
    print(ds.groupby(["drone", "kind", "session"]).size())
