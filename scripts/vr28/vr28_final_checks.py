# -*- coding: utf-8 -*-
"""
VR28 final checks: redundancy rate 0.0075, HHI-entropy correlations,
segment-level spot checks, arithmetic identities.
"""
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from scipy import stats as sps

BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"

df = pd.read_csv(TRIP)
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
res = {}
N = 87105

# ---------- item redundancy rate 0.0075 ----------
# Definition (Table 7 note): mean consumer-level share of purchased items
# transaction-linked to more than one patronized agent.
# Operational reading fixed by exact match of 342: item purchased from >=2 of the consumer's own agents.
multi_users = set(six.loc[six["n_agents"] >= 2, "user_id"].tolist())
multi_df = df[df["user_id"].isin(multi_users)]

# per consumer: items purchased from k>=2 own agents
ui_counts = multi_df.groupby(["user_id", "item_id"])["agent_id"].nunique().reset_index(name="k")
# distinct items per consumer
items_per_u = multi_df.groupby("user_id")["item_id"].nunique()
red_items = ui_counts[ui_counts["k"] >= 2].groupby("user_id").size()
red_rate = (red_items.reindex(items_per_u.index).fillna(0) / items_per_u)
mean_red_A = float(red_rate.mean())
# path B: numpy-ish independent recount via pivot of (u,item,agent) distinct
piv = multi_df.drop_duplicates(["user_id", "item_id", "agent_id"]).groupby(["user_id", "item_id"]).size()
red_items_B = piv[piv >= 2].groupby("user_id").size()
red_rate_B = (red_items_B.reindex(items_per_u.index).fillna(0) / items_per_u)
mean_red_B = float(red_rate_B.mean())
assert abs(mean_red_A - mean_red_B) < 1e-15
res["item_redundancy_rate_mean"] = mean_red_A
res["item_redundancy_rate_mean_4dp"] = round(mean_red_A, 4)
# incidence under this reading (should be 342)
res["redundancy_incidence_consumers"] = int((red_rate > 0).sum())
res["redundancy_incidence_pct"] = round(100.0 * float((red_rate > 0).mean()), 4)

# ---------- HHI vs Entropy correlations (N=87,105) ----------
H = six["HHI"].to_numpy(); E = six["Entropy"].to_numpy()
pr, pp = sps.pearsonr(H, E)
sr, sp = sps.spearmanr(H, E)
res["pearson_hhi_entropy"] = round(float(pr), 4)
res["pearson_p"] = float(pp)
res["spearman_hhi_entropy"] = round(float(sr), 4)
res["spearman_p"] = float(sp)

# ---------- segment-level spot checks (segment = item_id // 10000) ----------
seg = (df["item_id"].to_numpy() // 10000)
df["_seg"] = seg
# consumer-level segment entropy
gseg = df.groupby(["user_id", "_seg"]).size().reset_index(name="c")
tot_u = gseg.groupby("user_id")["c"].transform("sum")
gseg["p"] = gseg["c"] / tot_u
ent_u = gseg.groupby("user_id").apply(lambda x: float(-(x["p"] * np.log(x["p"])).sum()), include_groups=False)
mean_seg_ent_A = float(ent_u.mean())
# path B: scipy entropy per consumer via crosstab normalized
ct = pd.crosstab(df["user_id"], df["_seg"])
pn = ct.div(ct.sum(axis=1), axis=0).to_numpy()
with np.errstate(divide="ignore", invalid="ignore"):
    lg = np.where(pn > 0, np.log(pn), 0.0)
ent_B = -(pn * lg).sum(axis=1)
mean_seg_ent_B = float(np.mean(ent_B))
assert abs(mean_seg_ent_A - mean_seg_ent_B) < 1e-9
res["mean_consumer_segment_entropy_nats"] = round(mean_seg_ent_A, 4)
res["seg_ent_over_ln8_pct"] = round(100.0 * mean_seg_ent_A / np.log(8), 4)
# single-segment buyers
nseg_u = df.groupby("user_id")["_seg"].nunique()
single_seg = int((nseg_u == 1).sum())
res["single_segment_buyers"] = single_seg
res["single_segment_pct"] = round(100.0 * single_seg / N, 4)
# platform aggregate segment entropy / ln(8)
plat = df["_seg"].value_counts(normalize=True).to_numpy()
plat_ent = float(-(plat * np.log(plat)).sum())
res["platform_segment_entropy_nats"] = round(plat_ent, 4)
res["platform_seg_over_ln8_pct"] = round(100.0 * plat_ent / np.log(8), 4)
# pair-level segment entropy / ln(8) over 101,577 pairs
gpair = df.groupby(["user_id", "agent_id", "_seg"]).size().reset_index(name="c")
tot_p = gpair.groupby(["user_id", "agent_id"])["c"].transform("sum")
gpair["p"] = gpair["c"] / tot_p
ent_p = gpair.groupby(["user_id", "agent_id"]).apply(lambda x: float(-(x["p"] * np.log(x["p"])).sum()), include_groups=False)
mean_pair_ent = float(ent_p.mean())
res["pair_segment_entropy_over_ln8_pct"] = round(100.0 * mean_pair_ent / np.log(8), 4)
res["pair_segment_entropy_pairs"] = int(len(ent_p))
# cross-level drops
res["drop_platform_to_consumer_pp"] = round(res["platform_seg_over_ln8_pct"] - res["seg_ent_over_ln8_pct"], 4)
res["drop_consumer_to_pair_pp"] = round(res["seg_ent_over_ln8_pct"] - res["pair_segment_entropy_over_ln8_pct"], 4)

# ---------- arithmetic identities ----------
res["id_34_over_86710_pct"] = round(100.0 * 34 / 86710, 4)         # 0.039%
res["id_181_plus_34"] = 181 + 34                                    # 215
res["id_342_over_11694_pct"] = round(100.0 * 342 / 11694, 4)        # 2.925
res["id_5911_over_11694_pct"] = round(100.0 * 5911 / 11694, 4)      # 50.5
res["id_zero_overlap_pct"] = round(100.0 * (11694 - 342) / 11694, 4)  # 97.1
res["id_686246_over_101577"] = round(686246 / 101577, 4)            # 6.75
res["id_686246_over_87105"] = round(686246 / 87105, 4)              # 7.88
res["id_2925_minus_2095_pp"] = round(2.925 - 2.095, 3)              # 0.830
res["id_3194_minus_2925_pp"] = round(3.194 - 2.925, 3)              # 0.269
res["id_0075_minus_0028"] = round(0.0075 - 0.0028, 4)               # 0.0047
res["id_0075_minus_0056"] = round(0.0075 - 0.0056, 4)               # 0.0019
res["id_11694_over_87105_pct"] = round(100.0 * 11694 / 87105, 4)    # 13.4252
res["id_937_over_87105_pct"] = round(100.0 * 937 / 87105, 4)        # 1.0752
res["id_7635_over_87105_pct"] = round(100.0 * 7635 / 87105, 4)      # 8.7653
res["id_79470_over_87105_pct"] = round(100.0 * 79470 / 87105, 4)    # 91.2347
res["id_75411_over_87105_pct"] = round(100.0 * 75411 / 87105, 4)    # 86.5748
res["id_14867_over_101577_pct"] = round(100.0 * 14867 / 101577, 4)  # 14.64
res["id_8572_minus_7635"] = 8572 - 7635                             # 937
res["id_75457_minus_75411"] = 75457 - 75411                         # 46
res["id_86710_minus_177"] = 86710 - 177                             # 86533

with open(BASE + "vr28_final_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
