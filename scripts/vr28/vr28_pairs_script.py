# -*- coding: utf-8 -*-
"""
VR28 pair-level audit from raw triplets (686,246 rows). Dual-path everywhere.
Path A: pandas groupby. Path B: numpy unique / lexicographic sort.
"""
import pandas as pd
import numpy as np
import json

BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"

df = pd.read_csv(TRIP)
res = {}
res["triplet_rows"] = int(len(df))

u = df["user_id"].to_numpy()
a = df["agent_id"].to_numpy()
it = df["item_id"].to_numpy()

# ---------- pair (user,agent) purchase counts ----------
# Path A: pandas groupby size
g_pd = df.groupby(["user_id", "agent_id"]).size()
total_pairs_A = int(len(g_pd))
g1_A = int((g_pd == 1).sum())
gge2_A = int((g_pd >= 2).sum())

# Path B: numpy structured unique
pair_key = np.stack([u, a], axis=1)
pair_key_sorted = pair_key[np.lexsort((a, u))]
# row-wise unique via void view
dt = np.dtype([("u", pair_key.dtype), ("a", pair_key.dtype)])
voids = np.ascontiguousarray(pair_key_sorted).view(dt).ravel()
uniq, counts = np.unique(voids, return_counts=True)
total_pairs_B = int(len(uniq))
g1_B = int(np.count_nonzero(counts == 1))
gge2_B = int(np.count_nonzero(counts >= 2))

assert total_pairs_A == total_pairs_B and g1_A == g1_B and gge2_A == gge2_B, \
    (total_pairs_A, total_pairs_B, g1_A, g1_B, gge2_A, gge2_B)
res["total_pairs"] = total_pairs_A
res["pairs_g_eq1"] = g1_A
res["pairs_g_ge2"] = gge2_A
res["g1_share_pct"] = round(100.0 * g1_A / total_pairs_A, 4)

# ---------- masked pairs: g>=2 AND agent's observed shelf has exactly 1 item ----------
# Path A: pandas - distinct items per agent
shelf_pd = df.groupby("agent_id")["item_id"].nunique()
shelf1_agents_pd = set(shelf_pd[shelf_pd == 1].index.tolist())
pairs_df = g_pd.reset_index(name="g")
masked_A = int(((pairs_df["g"] >= 2) & (pairs_df["agent_id"].isin(shelf1_agents_pd))).sum())

# Path B: numpy - distinct (agent,item) pairs, count per agent
ai = np.stack([a, it], axis=1)
ai_sorted = ai[np.lexsort((it, a))]
dt2 = np.dtype([("a", ai.dtype), ("i", ai.dtype)])
ai_void = np.ascontiguousarray(ai_sorted).view(dt2).ravel()
ai_uniq = np.unique(ai_void)
agents_of_ai = ai_uniq["a"]
ag_uniq, ag_counts = np.unique(agents_of_ai, return_counts=True)
shelf1_agents_np = set(ag_uniq[ag_counts == 1].tolist())
# mask on the numpy pair table
pair_agents = uniq["a"]
pair_g = counts
masked_B = int(np.count_nonzero((pair_g >= 2) & np.isin(pair_agents, list(shelf1_agents_np))))

assert masked_A == masked_B, (masked_A, masked_B)
assert shelf1_agents_pd == shelf1_agents_np
res["masked_pairs"] = masked_A
res["gge2_minus_masked"] = gge2_A - masked_A
res["shelf1_agents_count"] = len(shelf1_agents_pd)

# ---------- unique consumers among g>=2 pairs ----------
cons_A = int(pairs_df.loc[pairs_df["g"] >= 2, "user_id"].nunique())
cons_B = int(len(np.unique(uniq["u"][pair_g >= 2])))
assert cons_A == cons_B
res["consumers_gge2"] = cons_A

# ---------- distinct purchased items ----------
items_A = int(df["item_id"].nunique())
items_B = int(len(np.unique(it)))
assert items_A == items_B
res["distinct_items"] = items_A

# ---------- distinct agents / users in triplets (context) ----------
res["distinct_agents"] = int(df["agent_id"].nunique())
res["distinct_users_in_triplets"] = int(df["user_id"].nunique())

with open(BASE + "vr28_pairs_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
