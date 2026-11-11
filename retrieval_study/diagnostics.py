"""
RS-1d diagnostics (pre-registered in context/decisions_retrieval.md), on one fold's validation queries. Descriptive.

(i)   Does a similar past mean a similar future? Key similarity vs value similarity over (query, random candidate)
      pairs, for the RAFT key (Pearson of the 14 days) and the R0 key (cosine of v1's 17 features).
(ii)  Can the key find the right analogues? The oracle analogues are the 5 candidates whose actual 5-day changes are
      closest to the query's. Where do the keys rank them, and how good would the oracle forecast be?
(iii) "Cold days": on heatwave days, how far below the truth are the R0 analogues' raw futures and RAFT's change-based
      forecast? Do the analogues start cooler than the query? Is anything in the pool as hot as the truth?
"""
from __future__ import annotations

import numpy as np

from retrieval_study import raft
from retrieval_study.testbed import FoldData

N_PAIRS = 200
N_ORACLE = 5
PAIR_SEED = 20261111


def _value_similarity(cand_change: np.ndarray, query_change: np.ndarray) -> np.ndarray:
    """(n, N) minus the RMSE between each query's actual 5-day change (n, F) and each candidate's (N, F), computed as
    |c|^2 + |q|^2 - 2 q.c so that no (n, N, F) array is built."""
    f = query_change.shape[1]
    sq = (np.sum(query_change ** 2, axis=1)[:, None] + np.sum(cand_change ** 2, axis=1)[None, :]
          - 2.0 * query_change @ cand_change.T) / f
    return -np.sqrt(np.maximum(sq, 0.0))


def _percentile_of(values: np.ndarray, chosen: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(n, k) share (%) of each row's eligible candidates whose value is lower than the chosen ones' (100 = highest)."""
    v = np.where(ok, values, np.nan)
    n_ok = ok.sum(axis=1, keepdims=True)
    picked = np.take_along_axis(values, chosen, axis=1)
    below = np.stack([np.nansum(v < picked[:, [j]], axis=1) for j in range(chosen.shape[1])], axis=1)
    return 100.0 * below / np.maximum(n_ok - 1, 1)


def fold_diagnostics(fd: FoldData, raft_m: int) -> dict:
    """Per-row arrays for one fold (pooled later): pair similarities for (i), oracle ranks and forecasts for (ii),
    heatwave-day biases for (iii). raft_m: RAFT's m chosen on B2 for this fold."""
    tr, va = fd.train, fd.val
    c_change = raft.offset_values(tr.future, tr.last)  # (N, 5)
    q_change = raft.offset_values(va.future, va.last)  # (n, 5)
    ok = raft.eligible(va.day, tr.day)  # (n, N)
    k_raft = (raft.standardise_rows(va.inputs) @ raft.standardise_rows(tr.inputs).T) / tr.inputs.shape[1]
    k_r0 = fd.fr.query_vectors(va.q) @ fd.fr.vectors.T
    v_sim = _value_similarity(c_change, q_change)  # (n, N)
    stratum = fd.d["stratum"].to_numpy()[va.pos[:, None] + np.arange(va.future.shape[1])[None, :]]
    hot_query = (stratum == "extreme").any(axis=1)

    # (i) random eligible pairs
    rng = np.random.default_rng([PAIR_SEED, int(fd.fold[1])])
    pairs = np.stack([rng.choice(np.flatnonzero(ok[i]), N_PAIRS, replace=False) for i in range(len(va.q))])
    pair = {"raft": np.take_along_axis(k_raft, pairs, axis=1), "r0": np.take_along_axis(k_r0, pairs, axis=1),
            "value": np.take_along_axis(v_sim, pairs, axis=1), "hot": np.repeat(hot_query[:, None], N_PAIRS, axis=1)}

    # (ii) oracle analogues: the N_ORACLE eligible candidates with the closest actual changes
    oracle = raft._top_k(np.where(ok, v_sim, -np.inf), N_ORACLE)
    rank = {"raft": _percentile_of(k_raft, oracle, ok), "r0": _percentile_of(k_r0, oracle, ok)}
    oracle_fc = va.last[:, None] + c_change[oracle].mean(axis=1)

    # real retrievals' own forecasts: RAFT change-based (p = 1, top raft_m), R0 level-based (mean raw future)
    nb = fd.raft_neighbours()[1]
    w = raft.weights(nb.sim[1], raft_m)
    safe = np.where(nb.idx[1][:, :raft_m] >= 0, nb.idx[1][:, :raft_m], 0)
    raft_fc = va.last[:, None] + np.einsum("nm,nmf->nf", w, c_change[safe])
    r0_idx = fd.r0_idx()[1]
    r0_fc = tr.future[r0_idx].mean(axis=1)  # raw levels
    r0_last_gap = va.last - tr.last[r0_idx].mean(axis=1)  # query's last value minus the analogues'

    # (iii) is any eligible candidate's raw future at that lead as hot as the truth?
    pool_max = np.stack([np.where(ok, tr.future[None, :, lead], -np.inf).max(axis=1)
                         for lead in range(va.future.shape[1])], axis=1)
    return {"pair": pair, "rank": rank, "hot_query": hot_query, "extreme": stratum == "extreme", "truth": va.future,
            "oracle_fc": oracle_fc, "raft_fc": raft_fc, "r0_fc": r0_fc, "r0_last_gap": r0_last_gap,
            "nothing_as_hot": pool_max < va.future}


def summarise(parts: list[dict]) -> dict:
    """Pool the folds' diagnostics into the report numbers."""
    from scipy.stats import spearmanr

    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("hot_query", "extreme", "truth", "oracle_fc", "raft_fc",
                                                               "r0_fc", "r0_last_gap", "nothing_as_hot")}
    pair = {k: np.concatenate([p["pair"][k] for p in parts]) for k in parts[0]["pair"]}
    rank = {k: np.concatenate([p["rank"][k] for p in parts]) for k in parts[0]["rank"]}
    out = {"key_value_spearman": {}, "oracle_rank_median": {}, "oracle_rank_mean": {}}
    for key in ("raft", "r0"):
        for sub, mask in (("all", np.ones_like(pair["hot"], bool)), ("heatwave queries", pair["hot"])):
            rho = spearmanr(pair[key][mask].ravel(), pair["value"][mask].ravel()).statistic
            out["key_value_spearman"][f"{key}, {sub}"] = float(rho)
        for sub, qmask in (("all", np.ones(len(cat["hot_query"]), bool)), ("heatwave queries", cat["hot_query"])):
            r = rank[key][qmask].ravel()
            out["oracle_rank_median"][f"{key}, {sub}"] = float(np.median(r))
            out["oracle_rank_mean"][f"{key}, {sub}"] = float(np.mean(r))

    def err_stats(fc: np.ndarray, mask: np.ndarray) -> dict:
        e = (fc - cat["truth"])[mask]
        return {"rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "n": int(mask.sum())}

    every = np.ones_like(cat["extreme"], bool)
    out["forecasts"] = {name: {"all": err_stats(cat[f"{name}_fc"], every), "heatwave days": err_stats(cat[f"{name}_fc"],
                                                                                                    cat["extreme"])}
                        for name in ("oracle", "raft", "r0")}
    ext = cat["extreme"]
    out["cold_days_by_lead"] = [{
        "lead": lead + 1, "rows": int(ext[:, lead].sum()),
        "r0_bias": float(np.mean((cat["r0_fc"] - cat["truth"])[ext[:, lead], lead])),
        "raft_bias": float(np.mean((cat["raft_fc"] - cat["truth"])[ext[:, lead], lead])),
        "nothing_as_hot_share": float(np.mean(cat["nothing_as_hot"][ext[:, lead], lead])),
    } for lead in range(ext.shape[1])]
    hot_rows = ext.any(axis=1)
    out["r0_last_gap"] = {"all queries": float(np.mean(cat["r0_last_gap"])),
                          "queries with a heatwave target day": float(np.mean(cat["r0_last_gap"][hot_rows]))}
    out["n_queries"] = int(len(cat["hot_query"]))
    out["n_heatwave_queries"] = int(cat["hot_query"].sum())
    return out
