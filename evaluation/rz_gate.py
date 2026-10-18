"""
Rz (500 hPa pattern retrieval) on C3-hop, WBGT only (it passed its screen only for WBGT), judged by the
rule written 2026-10-17 before any z500 data was downloaded: G3's three conditions against A2Lr_hw5_C3hop
(all days not worse; extreme days better) and against A2Lr_hw5_C3hop_rand (extreme days better).
Descriptive: per-lead deltas, attention concentration, and Rz vs Rg on the same backbone.

Run from repo root:  python -m evaluation.rz_gate   ->  evaluation_v2/rz_gate.{md,json}
"""
from __future__ import annotations

import json

from evaluation.compare_v2 import compare
from evaluation.gr_star import _need, _t, attention_stats, gate_rows, load_analogues, per_lead
from evaluation.retrieval_information_check import OUT_DIR

FAMILY = {"WBGT (physical)": "A2Lr_hw5"}


def main() -> None:
    (fam, base), = FAMILY.items()
    g = gate_rows("Rz", FAMILY)[fam]
    rz, rg = _need(f"{base}_C3hop_Rz"), _need(f"{base}_C3hop_Rg")
    diag = {"per_lead": per_lead(base, "Rz"), "attention_rz": attention_stats(load_analogues(f"{base}_C3hop_Rz")),
            "attention_rg": attention_stats(load_analogues(f"{base}_C3hop_Rg")),
            "rz_vs_rg_all": compare(rg, rz, "all"), "rz_vs_rg_ext": compare(rg, rz, "extreme")}
    L = ["# Rz on C3-hop, WBGT (rule written 2026-10-17, before any z500 data)", "",
         "Out of fold 2007-2018, 10 seeds x 4 folds, paired cluster-jackknife; Δ = C3hop_Rz minus comparison "
         "(negative = Rz better).", "",
         "| RMSE all: Rz / C3-hop | Extreme: Rz / C3-hop | (1) vs C3-hop, all | (2) vs C3-hop, extreme | "
         "(3) vs random, extreme | Result | Final test model |", "|---|---|---|---|---|---|---|",
         f"| {g['rmse_all']:.3f} / {g['ctrl_rmse_all']:.3f} | {g['rmse_extreme']:.3f} / {g['ctrl_rmse_extreme']:.3f} | "
         f"{_t(g['ctrl_all'])} | {_t(g['ctrl_ext'])} | {_t(g['rand_ext'])} | {'PASS' if g['passes'] else 'fail'} "
         f"{g['g3']} | `{g['final_model']}` |", "",
         f"Rz vs random, all days: {_t(g['rand_all'])}. Extreme-day detectable effect: {g['ext_mde']:.2f} °C. "
         f"Extreme bias Rz / C3-hop: {g['ext_bias']:+.2f} / {g['ctrl_ext_bias']:+.2f}.", "",
         "## Descriptive", "",
         "**Per lead (all days, Rz - C3-hop):** " + "; ".join(f"day {i} {_t(r)}" for i, r in enumerate(diag["per_lead"], 1)),
         "", f"**Rz vs Rg (same backbone):** all days {_t(diag['rz_vs_rg_all'])}; extreme {_t(diag['rz_vs_rg_ext'])}", "",
         f"**Attention:** Rz max weight {diag['attention_rz']['mean_max_weight']:.3f}, normalised entropy "
         f"{diag['attention_rz']['mean_norm_entropy']:.3f}; Rg {diag['attention_rg']['mean_max_weight']:.3f} / "
         f"{diag['attention_rg']['mean_norm_entropy']:.3f} (uniform: entropy 1)."]
    (OUT_DIR / "rz_gate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT_DIR / "rz_gate.json").write_text(json.dumps({"gate": g, "diagnostics": diag}, indent=2, default=float) + "\n",
                                          encoding="utf-8")
    print({fam: (g["passes"], g["final_model"])})


if __name__ == "__main__":
    main()
