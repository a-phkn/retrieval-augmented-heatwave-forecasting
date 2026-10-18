"""
What the graph learned (option 3; specified 2026-10-18 in context/decisions.md before it was computed;
descriptive). Occlusion on the trained C3-hop checkpoints (10 seeds x 4 folds, validation windows 2007-2018):
an input is "switched off" by replacing it with its training mean (0 after z-scoring), and the rise in RMSE
is its importance.

Scenarios: each of the 27 upstream points on all 14 input days, and on the last input day only; all points
at hop distance k from Delhi (k = 1..4) together; Delhi's own inputs. Reported per family as tables and a map
(evaluation_v2/figures/graph_importance.png). Claims stop at "the forecasts depend on these inputs".

Run from repo root:  python -m evaluation.graph_explain   ->  evaluation_v2/graph_explain.{md,json} + figure
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import training.train_unified as tu
from evaluation.retrieval_information_check import OUT_DIR
from models.dstgnn_multihop import hop_distances
from pipeline.graph import node_coords
from training.folds import FOLDS, build_fold
from training.graph_data import NODE_IDS, build_upstream, load_upstream

FAMILIES = {"Tmax": "A1prime_hw5_C3hop", "WBGT (physical)": "A2Lr_hw5_C3hop"}
BATCH = 512


def predict(model, X: np.ndarray, x_up: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), BATCH):
            out.append(model(torch.from_numpy(X[s:s + BATCH]), torch.from_numpy(x_up[s:s + BATCH])).numpy())
    return np.concatenate(out)


def scenarios(n_up: int, hops: np.ndarray) -> dict[str, tuple[str, list[int], bool]]:
    """name -> (kind, upstream node indices 0..26, last day only). kind 'up' or 'delhi'."""
    s = {"Delhi inputs": ("delhi", [], False)}
    for j in range(n_up):
        s[f"{NODE_IDS[j]} all days"] = ("up", [j], False)
        s[f"{NODE_IDS[j]} last day"] = ("up", [j], True)
    for k in sorted(set(hops.tolist())):
        s[f"hop {k} (all days)"] = ("up", [j for j in range(n_up) if hops[j] == k], False)
    return s


def rmse(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray | None = None) -> float:
    e = (pred - actual) if mask is None else (pred - actual)[mask]
    return float(np.sqrt(np.mean(e ** 2)))


def family(rid: str, up_table: pd.DataFrame, folds=tuple(FOLDS), seeds=range(10)) -> pd.DataFrame:
    cfg = tu.load_config(tu.REPO_ROOT / "configs" / f"{rid}.json")
    rows = []
    for fold in folds:
        data = build_fold(fold, cfg["target"], cfg["labels"], target_form=cfg["target_form"])
        upd = build_upstream(fold, up_table)
        val = data.val
        x_up = upd.x[upd.positions(val.query_dates)]  # (N, 14, 27, F)
        hops = hop_distances(upd.mask)[1:, 0]  # each upstream point's distance to Delhi
        ext = val.stratum == "extreme"
        saved = pd.read_parquet(tu.PRED_DIR / rid / f"{fold}.parquet")
        for seed in seeds:
            model = tu.make_dstgnn(cfg["graph"], val.X.shape[2], upd, cfg.get("hparams"))
            ckpt = tu.MODEL_DIR / rid / fold / f"seed_{seed}" / "checkpoint.pt"
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            base = data.to_raw(predict(model, val.X, x_up), "val")
            ref = saved[saved["seed"] == seed].sort_values(["query_date", "lead"])["pred"].to_numpy()
            if not np.allclose(base.reshape(-1), ref, atol=1e-4):
                raise ValueError(f"{rid} {fold} seed {seed}: rebuilt checkpoint does not reproduce the saved forecasts")
            b_all, b_ext = rmse(base, val.y_raw), rmse(base, val.y_raw, ext)
            for name, (kind, nodes, last) in scenarios(x_up.shape[2], hops).items():
                X, xu = val.X, x_up
                if kind == "delhi":
                    X = np.zeros_like(val.X)
                else:
                    xu = x_up.copy()
                    if last:
                        xu[:, -1, nodes] = 0.0
                    else:
                        xu[:, :, nodes] = 0.0
                p = data.to_raw(predict(model, X, xu), "val")
                rows.append({"fold": fold, "seed": seed, "scenario": name, "d_all": rmse(p, val.y_raw) - b_all,
                             "d_ext": rmse(p, val.y_raw, ext) - b_ext})
        print(f"{rid} {fold} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("scenario")[["d_all", "d_ext"]]
    return g.mean().join(g.std().rename(columns={"d_all": "sd_all", "d_ext": "sd_ext"})).sort_values("d_all", ascending=False)


def draw_map(res: dict[str, pd.DataFrame], path) -> None:
    """Two rows (all days, extreme days) x one column per family: RMSE rise when each point is switched off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = node_coords()
    fig, axes = plt.subplots(2, len(res), figsize=(6 * len(res), 9.5), constrained_layout=True, squeeze=False)
    for col, (fam, s) in enumerate(res.items()):
        for row, (key, label) in enumerate((("d_all", "all days"), ("d_ext", "heatwave (extreme) days"))):
            ax = axes[row, col]
            vals = np.array([s.loc[f"{n} all days", key] for n in NODE_IDS])
            lim = np.abs(vals).max()
            sc = ax.scatter(coords[1:, 1], coords[1:, 0], c=vals, s=80, cmap="RdBu_r", vmin=-lim, vmax=lim,
                            edgecolors="k", linewidths=0.4)
            ax.scatter([coords[0, 1]], [coords[0, 0]], marker="*", s=240, c="gold", edgecolors="k", label="Delhi")
            ax.set(title=f"{fam}, {label}", xlabel="longitude (°E)", ylabel="latitude (°N)")
            ax.legend(loc="lower right")
            fig.colorbar(sc, ax=ax, shrink=0.8, label="RMSE rise when switched off (°C)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def redraw() -> None:
    """Redraw the map from evaluation_v2/graph_explain.json (no recomputation)."""
    data = json.loads((OUT_DIR / "graph_explain.json").read_text(encoding="utf-8"))
    res = {fam: pd.DataFrame(rows).set_index("scenario") for fam, rows in data.items()}
    draw_map(res, OUT_DIR / "figures" / "graph_importance.png")


def write_report(res: dict[str, pd.DataFrame]) -> None:
    L = ["# What the graph learned: occlusion on C3-hop (specified 2026-10-18, before it was computed)", "",
         "Each input is replaced by its training mean on the validation windows (2007-2018, 10 seeds x 4 folds); the "
         "table gives the rise in RMSE (°C; mean over seeds and folds, SD in brackets). Larger = the forecast leans on "
         "it more. Descriptive only.", "", "![importance map](figures/graph_importance.png)", ""]
    for fam, s in res.items():
        top = s[~s.index.str.endswith("last day")]
        L += [f"## {fam}", "", "| Switched off | Δ RMSE all days | Δ RMSE extreme days |", "|---|---|---|"]
        for name, r in top.iterrows():
            L.append(f"| {name} | {r['d_all']:+.3f} ({r['sd_all']:.3f}) | {r['d_ext']:+.3f} ({r['sd_ext']:.3f}) |")
        last = s[s.index.str.endswith("last day")].head(8)
        L += ["", "Top 8 points, last input day only: " + "; ".join(f"{n.split()[0]} {r['d_all']:+.3f}"
                                                                   for n, r in last.iterrows()), ""]
    (OUT_DIR / "graph_explain.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    up_table = load_upstream()
    raw = {fam: family(rid, up_table) for fam, rid in FAMILIES.items()}
    res = {fam: summarise(df) for fam, df in raw.items()}
    draw_map(res, OUT_DIR / "figures" / "graph_importance.png")
    write_report(res)
    (OUT_DIR / "graph_explain.json").write_text(json.dumps({f: s.reset_index().to_dict(orient="records") for f, s in res.items()},
                                                           indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
