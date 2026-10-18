"""evaluation/graph_explain.py: the occlusion scenario list (no checkpoints needed)."""
import numpy as np

import evaluation.graph_explain as ge
from models.dstgnn_multihop import hop_distances
from pipeline.graph import neighbour_mask, node_coords


def test_scenarios_cover_every_point_twice_each_hop_ring_and_delhi():
    hops = hop_distances(neighbour_mask(node_coords()))[1:, 0]
    s = ge.scenarios(27, hops)
    assert len(s) == 1 + 2 * 27 + 4
    assert s["Delhi inputs"] == ("delhi", [], False)
    rings = [v[1] for k, v in s.items() if k.startswith("hop")]
    assert sorted(sum(rings, [])) == list(range(27))  # every point in exactly one ring
    assert all(s[f"{ge.NODE_IDS[j]} last day"][2] for j in range(27))
    assert ge.rmse(np.array([1.0, 3.0]), np.array([0.0, 0.0])) == np.sqrt(5.0)
