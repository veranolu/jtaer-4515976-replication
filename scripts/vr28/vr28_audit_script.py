# -*- coding: utf-8 -*-
"""
VR28 closure audit - dual-path verification for every count.
Path A: pandas boolean indexing.
Path B: numpy searchsorted / digitize / independent grouping.
All counts recomputed from raw detail files. No reverse-engineering from old figures.
"""
import pandas as pd
import numpy as np
import json

BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"

results = {}

# ---------------- Load user-level detail ----------------
fig2 = pd.read_csv(BASE + "fig2_user_hhi_topshare_87105_20260906.csv")
six  = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
hhi = fig2["hhi"].to_numpy(dtype=np.float64)
ts  = fig2["top_share"].to_numpy(dtype=np.float64)
N = len(fig2)
results["N_users"] = N
assert N == 87105, f"row count {N} != 87105"
assert len(six) == 87105

# sanity: six-var HHI/TopShare consistent with fig2 file
six_sorted = six.sort_values("user_id").reset_index(drop=True)
fig2_sorted = fig2.sort_values("user_id").reset_index(drop=True)
hhi_match = np.allclose(six_sorted["HHI"].to_numpy(), fig2_sorted["hhi"].to_numpy())
ts_match  = np.allclose(six_sorted["TopShare"].to_numpy(), fig2_sorted["top_share"].to_numpy())
results["sixvar_vs_fig2_hhi_identical"] = bool(hhi_match)
results["sixvar_vs_fig2_topshare_identical"] = bool(ts_match)

# ---------------- Deliverable 1: 21-row bins ----------------
edges = np.round(np.arange(0.0, 1.0, 0.05), 10)  # 0.00 ... 0.95  (20 edges -> 19 full bins + last-left 0.95)
# 19 continuous bins: [edges[i], edges[i+1]) for i=0..18 ; 20th bin [0.95,1.00); atom at exactly 1.00
rows = []
# Path A: boolean counts
bool_counts = []
for i in range(19):
    l, r = edges[i], edges[i+1]
    c = int(((hhi >= l) & (hhi < r)).sum())
    bool_counts.append(c)
    rows.append((l, r, c, "continuous"))
c20_bool = int(((hhi >= 0.95) & (hhi < 1.0)).sum())
atom_bool = int((hhi == 1.0).sum())
rows.append((0.95, 1.00, c20_bool, "continuous"))
rows.append((1.00, 1.00, atom_bool, "atom"))

# Path B: numpy searchsorted on sorted array
hhi_sorted = np.sort(hhi)
ss_counts = []
for i in range(19):
    l, r = edges[i], edges[i+1]
    c = int(np.searchsorted(hhi_sorted, r, side="left") - np.searchsorted(hhi_sorted, l, side="left"))
    ss_counts.append(c)
c20_ss = int(np.searchsorted(hhi_sorted, 1.0, side="left") - np.searchsorted(hhi_sorted, 0.95, side="left"))
atom_ss = int(len(hhi_sorted) - np.searchsorted(hhi_sorted, 1.0, side="left"))

# Path C (extra): np.digitize
bin_idx = np.digitize(hhi, edges[1:], right=False)  # 0..19 ; 19 means >=0.95
dig_counts = [int((bin_idx == i).sum()) for i in range(19)]
c20_dig = int(((bin_idx == 19) & (hhi < 1.0)).sum())
atom_dig = int(((bin_idx == 19) & (hhi == 1.0)).sum())

dual_ok = (bool_counts == ss_counts == dig_counts) and (c20_bool == c20_ss == c20_dig) and (atom_bool == atom_ss == atom_dig)
results["bins_dual_path_consistent"] = bool(dual_ok)
results["bins_bool"] = bool_counts
results["bins_searchsorted"] = ss_counts
results["bins_digitize"] = dig_counts
results["bin20_095_100_excl"] = c20_bool
results["atom_hhi_eq1"] = atom_bool
results["bins_total"] = int(sum(bool_counts) + c20_bool + atom_bool)

assert c20_bool == 46, f"bin20={c20_bool} != 46"
assert atom_bool == 75411, f"atom={atom_bool} != 75411"
assert sum(bool_counts) + c20_bool + atom_bool == 87105

# write deliverable 1
out1 = BASE + "fig2a_hhi_bins_final_20260907.csv"
with open(out1, "w", encoding="utf-8") as f:
    f.write("bin_left,bin_right,count,type\n")
    for l, r, c, t in rows:
        f.write(f"{l:.2f},{r:.2f},{c},{t}\n")
results["deliverable1_path"] = out1

# extra: count(0.99 <= hhi < 1.00) dual path
c99_bool = int(((hhi >= 0.99) & (hhi < 1.0)).sum())
c99_ss = int(np.searchsorted(hhi_sorted, 1.0, side="left") - np.searchsorted(hhi_sorted, 0.99, side="left"))
assert c99_bool == c99_ss
results["count_099_le_hhi_lt_100"] = c99_bool
# also the [0.95,0.99) piece for the boundary-disambiguation note
c9599_bool = int(((hhi >= 0.95) & (hhi < 0.99)).sum())
c9599_ss = int(np.searchsorted(hhi_sorted, 0.99, side="left") - np.searchsorted(hhi_sorted, 0.95, side="left"))
assert c9599_bool == c9599_ss
results["count_095_le_hhi_lt_099"] = c9599_bool

# ---------------- Core user-level audit numbers ----------------
def pct(x, nd=4):
    return round(100.0 * x / N, nd)

# top_share thresholds (dual path: pandas boolean vs numpy count_nonzero on same mask built two ways)
ts_lt_08_A = int((fig2["top_share"] < 0.80).sum())
ts_lt_08_B = int(np.count_nonzero(np.less(ts, 0.80)))
ts_ge_08_A = int((fig2["top_share"] >= 0.80).sum())
ts_ge_08_B = int(np.count_nonzero(np.greater_equal(ts, 0.80)))
ts_eq_08_A = int((fig2["top_share"] == 0.80).sum())
ts_eq_08_B = int(np.count_nonzero(np.equal(ts, 0.80)))
ts_le_08_A = int((fig2["top_share"] <= 0.80).sum())
ts_le_08_B = int(np.count_nonzero(np.less_equal(ts, 0.80)))
assert ts_lt_08_A == ts_lt_08_B and ts_ge_08_A == ts_ge_08_B and ts_eq_08_A == ts_eq_08_B and ts_le_08_A == ts_le_08_B
results["ts_lt_080"] = ts_lt_08_A
results["ts_ge_080"] = ts_ge_08_A
results["ts_eq_080"] = ts_eq_08_A
results["ts_le_080"] = ts_le_08_A
results["ts_le_minus_lt"] = ts_le_08_A - ts_lt_08_A  # should equal ts_eq_080
assert ts_le_08_A - ts_lt_08_A == ts_eq_08_A

hhi_eq1_A = int((fig2["hhi"] == 1.0).sum())
hhi_eq1_B = atom_ss
assert hhi_eq1_A == hhi_eq1_B
results["hhi_eq1"] = hhi_eq1_A

# multi-agent consumers from six-var file
na = six["n_agents"].to_numpy()
multi_A = int((six["n_agents"] >= 2).sum())
multi_B = int(np.count_nonzero(na >= 2))
assert multi_A == multi_B
results["multi_agent_consumers"] = multi_A

# multi-agent HHI mean / median (dual path: pandas vs numpy on independent mask construction)
mask_multi_pd = six["n_agents"] >= 2
hhi_multi_pd = six.loc[mask_multi_pd, "HHI"]
mean_A = float(hhi_multi_pd.mean()); median_A = float(hhi_multi_pd.median())
hhi6 = six["HHI"].to_numpy()
mask_multi_np = np.where(na >= 2)[0]
mean_B = float(np.mean(hhi6[mask_multi_np])); median_B = float(np.median(hhi6[mask_multi_np]))
assert abs(mean_A - mean_B) < 1e-12 and abs(median_A - median_B) < 1e-12
results["multi_agent_hhi_mean"] = mean_A
results["multi_agent_hhi_median"] = median_B
results["multi_agent_hhi_mean_3dp"] = round(mean_A, 3)
results["multi_agent_hhi_median_3dp"] = round(median_B, 3)

with open(BASE + "vr28_partial_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
