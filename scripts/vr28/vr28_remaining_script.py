# -*- coding: utf-8 -*-
"""
VR28 remaining verifiable numbers. Dual-path for every count.
"""
import pandas as pd
import numpy as np
import json
from collections import defaultdict

BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"

df = pd.read_csv(TRIP)
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
res = {}
N = 87105

# ---------- HHI thresholds (dual path) ----------
hhi6 = six["HHI"].to_numpy()
for thr, name in [(0.80, "hhi_gt_080"), (0.90, "hhi_gt_090")]:
    cA = int((six["HHI"] > thr).sum())
    cB = int(np.count_nonzero(hhi6 > thr))
    assert cA == cB
    res[name] = cA
    res[name + "_pct"] = round(100.0 * cA / N, 4)
med_hhi_A = float(six["HHI"].median()); med_hhi_B = float(np.median(hhi6))
assert med_hhi_A == med_hhi_B
res["median_hhi_all"] = med_hhi_A

# ---------- n_agents stats (dual path) ----------
na = six["n_agents"].to_numpy()
res["n_agents_mean"] = round(float(np.mean(na)), 4)
res["n_agents_mean_pd"] = round(float(six["n_agents"].mean()), 4)
res["n_agents_sd_sample"] = round(float(six["n_agents"].std()), 4)          # pandas default ddof=1
res["n_agents_sd_pop"] = round(float(np.std(na)), 4)                        # numpy default ddof=0
res["n_agents_median"] = float(np.median(na))
res["n_agents_max"] = int(na.max())

# ---------- n_purchases median: six-var vs triplets-derived ----------
npc6 = six["n_purchases"].to_numpy()
res["n_purchases_median_sixvar"] = float(np.median(npc6))
res["n_purchases_sum_sixvar"] = int(npc6.sum())
# from triplets
counts_trip = df.groupby("user_id").size()
res["trips_users"] = int(len(counts_trip))
# align: all 87105 users present?
assert len(counts_trip) == 87105
ct = counts_trip.to_numpy()
res["n_purchases_median_triplets"] = float(np.median(ct))
res["n_purchases_mean_triplets"] = round(float(ct.mean()), 4)
# value counts around 5/6
vc = np.bincount(ct)
res["count_purchases_le4"] = int(vc[:5].sum())
res["count_purchases_eq5"] = int(vc[5]) if len(vc) > 5 else 0
res["count_purchases_eq6"] = int(vc[6]) if len(vc) > 6 else 0
res["count_purchases_le5"] = int(vc[:6].sum())
# numpy independent median via sort
cts = np.sort(ct)
res["median_via_sort"] = float(cts[len(cts)//2]) if len(cts) % 2 == 1 else float((cts[len(cts)//2 - 1] + cts[len(cts)//2]) / 2)
# cross-check six-var n_purchases equals triplets counts per user
six_idx = six.set_index("user_id")["n_purchases"]
align = six_idx.reindex(counts_trip.index).to_numpy()
res["sixvar_npurch_eq_trips"] = bool(np.array_equal(align, ct))

# ---------- per-agent shelf stats (dual path) ----------
shelf_pd = df.groupby("agent_id")["item_id"].nunique()
res["shelf_mean"] = round(float(shelf_pd.mean()), 4)
res["shelf_median"] = float(shelf_pd.median())
# numpy path: unique (agent,item) then count per agent
ai = df[["agent_id", "item_id"]].drop_duplicates()
shelf_pd2 = ai.groupby("agent_id").size()
res["shelf_mean_path2"] = round(float(shelf_pd2.mean()), 4)
res["shelf_median_path2"] = float(shelf_pd2.median())
res["n_agents_with_sales"] = int(len(shelf_pd))
res["total_agent_item_pairs"] = int(len(ai))

# ---------- agent purchase volume concentration (dual path) ----------
av = df.groupby("agent_id").size().sort_values(ascending=False).to_numpy()
# numpy path
av2 = np.sort(df["agent_id"].value_counts().to_numpy())[::-1]
assert np.array_equal(av, av2)
tot = 686246
res["agent_max_purchases"] = int(av[0])
res["agent_top1_share_pct"] = round(100.0 * av[0] / tot, 4)
res["agent_top5_share_pct"] = round(100.0 * av[:5].sum() / tot, 4)
res["agent_top10_share_pct"] = round(100.0 * av[:10].sum() / tot, 4)
shares = av / tot
res["platform_agent_hhi"] = float(np.sum(shares ** 2))
res["agent_mean_purchases"] = round(tot / len(av), 4)

# ---------- split sum arithmetic ----------
res["split_sum"] = 553744 + 66231 + 66271

# ---------- pairs of multi-agent consumers (26,166?) ----------
pairs = df.groupby(["user_id", "agent_id"]).size().reset_index(name="g")
multi_users = set(six.loc[six["n_agents"] >= 2, "user_id"].tolist())
pmA = int(pairs["user_id"].isin(multi_users).sum())
# numpy path
pu = pairs["user_id"].to_numpy()
pmB = int(np.count_nonzero(np.isin(pu, list(multi_users))))
assert pmA == pmB
res["pairs_of_multi_agent_consumers"] = pmA

# ---------- cross-agent overlap among multi-agent consumers ----------
# shelf per agent (global, from full triplets)
agent_shelf = df.groupby("agent_id")["item_id"].apply(set).to_dict()
multi_df = df[df["user_id"].isin(multi_users)]
# purchases per (user, agent): item sets
ua_items = multi_df.groupby(["user_id", "agent_id"])["item_id"].apply(set)

user_agents = defaultdict(list)
for (u, a), items in ua_items.items():
    user_agents[u].append(a)

overlap_purchase = 0      # variant A: purchased item also transaction-linked to another of own agents
overlap_direct = 0        # variant B: same item purchased from >=2 of own agents
shelf_shared = 0          # any item shared between shelves of own agents (pairwise)
redund_flags = 0          # purchases whose item is in shelf of another own agent
tot_multi_purchases = 0

# precompute consumer purchase items per agent in dict form
ua_dict = {k: v for k, v in ua_items.items()}
for u, agents in user_agents.items():
    asets = [ua_dict[(u, a)] for a in agents]
    shelves = [agent_shelf[a] for a in agents]
    all_purch = set().union(*asets)
    tot_multi_purchases += sum(len(s) for s in asets)
    # variant B: direct repurchase of same item across own agents
    if len(agents) >= 2:
        inter_direct = set()
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                inter_direct |= (asets[i] & asets[j])
        if inter_direct:
            overlap_direct += 1
    # variant A: any purchased item in another own agent's shelf
    flagA = False
    for i, a in enumerate(agents):
        others_shelf = set()
        for j, b in enumerate(agents):
            if j != i:
                others_shelf |= shelves[j]
        if asets[i] & others_shelf:
            flagA = True
            redund_flags += len(asets[i] & others_shelf)
    if flagA:
        overlap_purchase += 1
    # shelf-level pairwise share
    flagshelf = False
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if shelves[i] & shelves[j]:
                flagshelf = True
                break
        if flagshelf:
            break
    if flagshelf:
        shelf_shared += 1

res["multi_agent_consumers_check"] = len(user_agents)
res["overlap_variantA_item_in_other_agent_shelf"] = overlap_purchase
res["overlap_variantA_pct"] = round(100.0 * overlap_purchase / len(user_agents), 4)
res["overlap_variantB_direct_item_repurchase"] = overlap_direct
res["overlap_variantB_pct"] = round(100.0 * overlap_direct / len(user_agents), 4)
res["shelf_level_shared_consumers"] = shelf_shared
res["shelf_level_shared_pct"] = round(100.0 * shelf_shared / len(user_agents), 4)
res["redundancy_flags"] = redund_flags
res["tot_multi_purchases"] = tot_multi_purchases
res["redundancy_rate_variantA"] = round(redund_flags / tot_multi_purchases, 6)

with open(BASE + "vr28_remaining_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
