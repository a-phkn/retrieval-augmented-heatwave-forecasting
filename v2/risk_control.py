"""
E5 (PROTOCOL_V2, H-C secondary arm): an "unsafe day tomorrow" trigger with a bounded miss rate, by
conformal risk control (Angelopoulos, Bates, Fisch, Lei & Schuster, "Conformal risk control", ICLR 2024).

Loss for a calibration unsafe day i at threshold lam: L_i(lam) = 1{score_i < lam} (missed), which is
monotone in lam and bounded by B = 1. CRC picks the largest lam with
    n/(n+1) * Rhat_n(lam) + B/(n+1) <= alpha,   i.e.  (k(lam) + 1) / (n + 1) <= alpha,
where n is the number of calibration unsafe days and k(lam) how many of them score below lam. Then, if the
calibration and new unsafe days are exchangeable, E[miss on a new unsafe day] <= alpha. Serial correlation
breaks exchangeability (PROTOCOL_V2 H-C known limit), so the realised miss rate is always reported.
The trigger fires when score >= lam.
"""
from __future__ import annotations

import numpy as np


def crc_threshold(scores_cal: np.ndarray, unsafe_cal: np.ndarray, alpha: float) -> float:
    """Largest threshold whose CRC bound on the miss rate is <= alpha. Returns -inf (always trigger)
    if even zero misses cannot meet the bound, i.e. 1/(n+1) > alpha."""
    pos = np.sort(np.asarray(scores_cal, float)[np.asarray(unsafe_cal, bool)])
    n = len(pos)
    k_max = int(np.floor(alpha * (n + 1) - 1 + 1e-12))  # largest k with (k+1)/(n+1) <= alpha
    if n == 0 or k_max < 0:
        return -np.inf
    if k_max >= n:
        return np.inf
    # lam may be any value in (pos[k_max-1], pos[k_max]]; take pos[k_max] so exactly k_max positives are below
    return float(pos[k_max])


def evaluate(scores: np.ndarray, unsafe: np.ndarray, lam: float) -> dict:
    s, y = np.asarray(scores, float), np.asarray(unsafe, bool)
    fire = s >= lam
    tp, fn, fp = int((fire & y).sum()), int((~fire & y).sum()), int((fire & ~y).sum())
    return {"threshold": lam, "n": int(len(s)), "unsafe_days": int(y.sum()), "triggers": int(fire.sum()),
            "miss_rate": fn / max(1, tp + fn), "false_trigger_ratio": fp / max(1, int(fire.sum())),
            "trigger_rate": float(fire.mean()) if len(s) else float("nan")}
