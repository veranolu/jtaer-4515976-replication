# -*- coding: utf-8 -*-
"""
Resolve 80,056 vs 79,429: unique consumers with g>=2.
Multiple interpretations, each dual-path.
"""
import pandas as pd
import numpy as np
import json

BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"

df = pd.read_csv(TRIP)
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
res = {}

u = df["user_id"].to_numpy()
a = df["agent_id"].to_numpy()
it = df["item_id"].to_numpy()

# --- Interpretation 1: consumers with at least one PAIR having g>=2 purchases ---
g_pd = df.groupby(["user_id", "agent_id"]).size()
pairs_df = g_pd.reset_index(name="g")
c1_A = int(pairs_df.loc[pairs_df["g"] >= 2, "user_id"].nunique())
# numpy path
key = np.stack([u, a], axis=1)
key_sorted = key[np.lexsort((a, u))]
dt = np.dtype([("u", key.dtype), ("a", key.dtype)])
voids = np.ascontiguousarray(key_sorted).view(dt).ravel()
uniq, counts = np.unique(voids, return_counts=True)
c1_B = int(len(np.unique(uniq["u"][counts >= 2])))
res["interp1_consumers_with_pair_gge2"] = (c1_A, c1_B)
assert c1_A == c1_B

# --- Interpretation 2: consumers whose TOTAL purchases >= 2 ---
npurch = six.set_index("user_id")["n_purchases"]
c2_A = int((npurch >= 2).sum())
c2_B = int(np.count_nonzero(six["n_purchases"].to_numpy() >= 2))
res["interp2_consumers_total_purchases_ge2"] = (c2_A, c2_B)
assert c2_A == c2_B

# --- Interpretation 3: consumers with at least one pair having >=2 DISTINCT items ---
di = df.groupby(["user_id", "agent_id"])["item_id"].nunique()
di_df = di.reset_index(name="d")
c3_A = int(di_df.loc[di_df["d"] >= 2, "user_id"].nunique())
# numpy path: distinct (u,a,i)
ui = np.stack([u, a, it], axis=1)
ui_sorted = ui[np.lexsort((it, a, u))]
dt3 = np.dtype([("u", ui.dtype), ("a", ui.dtype), ("i", ui.dtype)])
uiv = np.ascontiguousarray(ui_sorted).view(dt3).ravel()
ui_uniq = np.unique(uiv)
# count distinct items per (u,a)
ua_of = np.stack([ui_uniq["u"], ui_uniq["a"]], axis=1)
ua_sorted = ua_of[np.lexsort((ua_of[:, 1], ua_of[:, 0]))]
dt4 = np.dtype([("u", ua_of.dtype), ("a", ua_of.dtype)])
uav = np.ascontiguousarray(ua_sorted).view(dt4).ravel()
ua_uniq, ua_counts = np.unique(uav, return_counts=True)
c3_B = int(len(np.unique(ua_uniq["u"][ua_counts >= 2])))
res["interp3_consumers_with_pair_distinct_items_ge2"] = (c3_A, c3_B)
assert c3_A == c3_B

# --- Interpretation 4: consumers in pairs used for evenness after masking (g>=2 & shelf>1) ---
shelf = df.groupby("agent_id")["item_id"].nunique()
shelf1 = set(shelf[shelf == 1].index.tolist())
pairs_df["masked"] = (pairs_df["g"] >= 2) & (pairs_df["agent_id"].isin(shelf1))
c4_A = int(pairs_df.loc[(pairs_df["g"] >= 2) & (~pairs_df["agent_id"].isin(shelf1)), "user_id"].nunique())
# numpy path
pair_agents = uniq["a"]; pair_users = uniq["u"]; pair_g = counts
mask_np = (pair_g >= 2) & (~np.isin(pair_agents, list(shelf1)))
c4_B = int(len(np.unique(pair_users[mask_np])))
res["interp4_consumers_evenness_after_mask"] = (c4_A, c4_B)
assert c4_A == c4_B

# --- consumers among g=1 pairs (context) ---
c5_A = int(pairs_df.loc[pairs_df["g"] == 1, "user_id"].nunique())
c5_B = int(len(np.unique(pair_users[pair_g == 1])))
res["consumers_with_pair_g_eq1"] = (c5_A, c5_B)
assert c5_A == c5_B

# overlap between interp1 and interp2 sets
set1 = set(pairs_df.loc[pairs_df["g"] >= 2, "user_id"].tolist())
set2 = set(six.loc[six["n_purchases"] >= 2, "user_id"].tolist())
res["interp2_minus_interp1"] = len(set2 - set1)
res["interp1_minus_interp2"] = len(set1 - set2)

# all-consumer HHI mean (0.947 claim) dual path
hhi_all_mean_A = float(six["HHI"].mean())
hhi_all_mean_B = float(np.mean(six["HHI"].to_numpy()))
assert abs(hhi_all_mean_A - hhi_all_mean_B) < 1e-15
res["all_consumer_hhi_mean"] = hhi_all_mean_A
res["all_consumer_hhi_mean_3dp"] = round(hhi_all_mean_A, 3)

# median consumer purchases (5?) dual path
med_A = float(six["n_purchases"].median())
med_B = float(np.median(six["n_purchases"].to_numpy()))
assert med_A == med_B
res["median_consumer_purchases"] = med_A

# 686246/101577 avg purchases per pair
res["avg_purchases_per_pair"] = 686246.0 / 101577.0

# ln(5)/ln(37)
res["ln5_ln37"] = float(np.log(5) / np.log(37))

with open(BASE + "vr28_consumers_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
