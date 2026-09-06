# -*- coding: utf-8 -*-
"""
VR28-final item 9: agent-level 78.5% vs consumer-pair 78.46% (Table S8 anchors
0.8997 / 61.8% / 78.5%; consumer-pair N=18,953: 0.8066 / 78.46% / 0.2287 / 0.8786;
Ruzicka 0.5471). Segment = item_id // 10000. seed not needed (exact enumeration).
"""
import pandas as pd
import numpy as np
import json, time, datetime
from math import comb

t0 = time.time()
SCRIPT = "vr28b_segments_20260906.py"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
df = pd.read_csv(TRIP)
print(f"[{SCRIPT}] {datetime.datetime.now()} input triplets={len(df):,}", flush=True)

df["seg"] = df["item_id"] // 10000
n_seg = int(df["seg"].nunique())
R = {"n_segments": n_seg, "seg_ids": sorted(df["seg"].unique().tolist())}

# ---------- agent segment masks + segment sales shares ----------
ag_seg = df.groupby("agent_id")["seg"].agg(lambda s: sorted(set(s)))
agents = np.sort(df["agent_id"].unique())
nA = len(agents)
aidx = {a: i for i, a in enumerate(agents)}
FULL = (1 << n_seg) - 1
masks = np.zeros(nA, dtype=np.int64)
for a, segs in ag_seg.items():
    m = 0
    for s in segs:
        m |= (1 << int(s))
    masks[aidx[a]] = m

# segment sales shares per agent (purchase counts)
asc = df.groupby(["agent_id", "seg"]).size().unstack(fill_value=0).reindex(agents).fillna(0)
asc = asc.reindex(columns=sorted(df["seg"].unique()), fill_value=0)
shares = (asc.T / asc.sum(axis=1)).T.to_numpy(dtype=np.float64)

R["n_agents"] = nA
R["agent_pairs_total"] = comb(nA, 2)
full_cnt = int((masks == FULL).sum())
R["agents_full_seg_count"] = full_cnt
R["agents_full_seg_pct"] = 100.0 * full_cnt / nA

# ---------- all-pairs Jaccard / identical share via mask grouping (dual path) ----------
# path A: group by mask
uniq_m, cnt_m = np.unique(masks, return_counts=True)
pop = {int(m): int(bin(int(m)).count("1")) for m in uniq_m}
jac_sum = 0.0; ident_pairs = 0; total_pairs = 0
for i, mi in enumerate(uniq_m):
    ci = int(cnt_m[i])
    # same-mask pairs
    total_pairs += ci * (ci - 1) // 2
    ident_pairs += ci * (ci - 1) // 2
    jac_sum += 1.0 * ci * (ci - 1) / 2
    for j in range(i + 1, len(uniq_m)):
        mj = uniq_m[j]
        cj = int(cnt_m[j])
        inter = bin(int(mi) & int(mj)).count("1")
        union = bin(int(mi) | int(mj)).count("1")
        jv = inter / union if union else 1.0
        jac_sum += jv * ci * cj
        total_pairs += ci * cj
mean_jac_A = jac_sum / total_pairs
ident_share_A = ident_pairs / total_pairs
# path B: chunked direct (sample-free, full enumeration in blocks)
B = 256
lut = np.array([bin(int(x)).count("1") for x in range(1 << n_seg)], dtype=np.float64)
jac_sum2 = 0.0; ident2 = 0
for s0 in range(0, nA, B):
    s1 = min(s0 + B, nA)
    mi = masks[s0:s1]
    inter = np.bitwise_and(mi[:, None], masks[s0 + 1:])
    union = np.bitwise_or(mi[:, None], masks[s0 + 1:])
    inter_c = lut[inter]
    union_c = lut[union]
    jv = np.where(union_c > 0, inter_c / np.maximum(union_c, 1), 1.0)
    gj = np.arange(s0 + 1, nA)
    gi = np.arange(s0, s1)
    keep = gi[:, None] < gj[None, :]
    jac_sum2 += float(jv[keep].sum())
    ident2 += int((mi[:, None] == masks[s0 + 1:])[keep].sum())
    del inter, union, inter_c, union_c, jv, keep
mean_jac_B = jac_sum2 / total_pairs
ident_share_B = ident2 / total_pairs
R["jaccard_mean_agentpairs"] = mean_jac_A
R["jaccard_mean_agentpairs_pathB"] = mean_jac_B
R["identical_set_share_pct"] = 100 * ident_share_A
R["identical_set_share_pct_pathB"] = 100 * ident_share_B
assert abs(mean_jac_A - mean_jac_B) < 1e-12 and abs(ident_share_A - ident_share_B) < 1e-15

# ---------- Ruzicka (share-weighted) all agent pairs, chunked ----------
ruz_sum = 0.0; le05 = 0
for s0 in range(0, nA, B):
    s1 = min(s0 + B, nA)
    Si = shares[s0:s1]
    for t0b in range(s0 + 1, nA, 2000):
        t1b = min(t0b + 2000, nA)
        Sj = shares[t0b:t1b]
        mn = np.minimum(Si[:, None, :], Sj[None, :, :]).sum(axis=2)
        mx = np.maximum(Si[:, None, :], Sj[None, :, :]).sum(axis=2)
        rv = np.where(mx > 0, mn / np.maximum(mx, 1e-300), 1.0)
        gj = np.arange(t0b, t1b); gi = np.arange(s0, s1)
        keep = gi[:, None] < gj[None, :]
        ruz_sum += float(rv[keep].sum())
        le05 += int((rv <= 0.5)[keep].sum())
        del mn, mx, rv, keep
    print(f"  ruzicka chunk {s0}/{nA} ({time.time()-t0:.0f}s)", flush=True)
R["ruzicka_mean"] = ruz_sum / total_pairs
R["ruzicka_le05_pct"] = 100 * le05 / total_pairs

# ---------- within-consumer agent pairs (N=18,953?) ----------
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
nag = six["n_agents"].to_numpy()
R["within_consumer_pairs_arith"] = int(sum(comb(int(k), 2) for k in nag if k >= 2))

ua = df.groupby(["user_id", "agent_id"])
multi_users = set(six.loc[six["n_agents"] >= 2, "user_id"].tolist())
user_agents = df[df["user_id"].isin(multi_users)].groupby("user_id")["agent_id"].agg(lambda s: sorted(set(s)))
ag_mask_map = {a: masks[aidx[a]] for a in agents}
ag_share_map = {a: shares[aidx[a]] for a in agents}
lut = np.array([bin(int(x)).count("1") for x in range(1 << n_seg)], dtype=np.int64)

# per-consumer segment purchase counts for redundancy
useg = df[df["user_id"].isin(multi_users)].groupby(["user_id", "seg"]).size()
useg_dict = {u: grp for u, grp in useg.groupby(level=0)}

n_cp = 0; jac_s = 0.0; cov_full = 0; both_full = 0; inter_nonempty = 0; peragent_full = 0
red_rates = []
red_rates_purchase = []
for u, alist in user_agents.items():
    m = len(alist)
    # agent pair stats
    for i in range(m):
        for j in range(i + 1, m):
            n_cp += 1
            mi, mj = ag_mask_map[alist[i]], ag_mask_map[alist[j]]
            inter = int(lut[mi & mj]); union = int(lut[mi | mj])
            jac_s += inter / union if union else 1.0
            if union == n_seg:
                cov_full += 1
            if inter > 0:
                inter_nonempty += 1
            if mi == FULL and mj == FULL:
                both_full += 1
    peragent_full += sum(1 for a in alist if ag_mask_map[a] == FULL)
    # redundancy: per-consumer purchased items in segments linked to >1 of own agents
    # segment offer map: for each segment, how many of the consumer's agents offer it
    seg_offer = {}
    for a in alist:
        for s in ag_seg[a]:
            seg_offer[s] = seg_offer.get(s, 0) + 1
    u_counts = useg_dict.get(u)
    if u_counts is not None:
        tot = int(u_counts.sum())
        red_purch = int(sum(int(c) for s, c in u_counts.items() if seg_offer.get(s, 0) > 1))
        red_rates_purchase.append(red_purch / tot)
R["within_consumer_pairs"] = n_cp
R["cp_jaccard_mean"] = jac_s / n_cp
R["cp_fullseg_union_pct"] = 100 * cov_full / n_cp
R["cp_both_full_pct"] = 100 * both_full / n_cp
R["cp_intersect_nonempty_pct"] = 100 * inter_nonempty / n_cp
R["cp_purchase_level_redundancy"] = float(np.mean(red_rates_purchase))
R["cp_consumers"] = len(user_agents)

# distinct-item-level segment redundancy variant (mean over consumers)
multi_df = df[df["user_id"].isin(multi_users)]
ui = multi_df.groupby(["user_id", "item_id"])["seg"].first().reset_index()
red2 = []
for u, grp in ui.groupby("user_id"):
    alist = user_agents[u]
    seg_offer = {}
    for a in alist:
        for s in ag_seg[a]:
            seg_offer[s] = seg_offer.get(s, 0) + 1
    items = grp["seg"].to_numpy()
    if len(items):
        red2.append(float(np.mean([1.0 if seg_offer.get(s, 0) > 1 else 0.0 for s in items])))
R["cp_item_level_seg_redundancy"] = float(np.mean(red2))

with open(BASE + "vr28b_segments_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
