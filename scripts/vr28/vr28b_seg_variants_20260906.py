# -*- coding: utf-8 -*-
"""
VR28-final item 9 round 2: variant battery to pin exact definitions of
consumer-pair (N=18,953) 78.46% full-segment coverage / 0.8786 shelf-level redundancy /
0.2287 purchase-level redundancy. Exact enumeration, no RNG.
"""
import pandas as pd
import numpy as np
import json, time, datetime

t0 = time.time()
SCRIPT = "vr28b_seg_variants_20260906.py"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
df = pd.read_csv(TRIP)
print(f"[{SCRIPT}] {datetime.datetime.now()} input triplets={len(df):,}", flush=True)
df["seg"] = df["item_id"] // 10000
n_seg = 8
FULL = (1 << n_seg) - 1
lut = np.array([bin(int(x)).count("1") for x in range(1 << n_seg)], dtype=np.int64)

six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
multi_users = set(six.loc[six["n_agents"] >= 2, "user_id"].tolist())
mdf = df[df["user_id"].isin(multi_users)]

ag_seg = df.groupby("agent_id")["seg"].agg(lambda s: frozenset(s))
ag_mask = {a: sum(1 << int(s) for s in ss) for a, ss in ag_seg.items()}
ag_nseg = {a: len(ss) for a, ss in ag_seg.items()}

user_agents = mdf.groupby("user_id")["agent_id"].agg(lambda s: sorted(set(s)))
# consumer's purchased segments per (user, agent)
upseg = mdf.groupby(["user_id", "agent_id"])["seg"].agg(lambda s: frozenset(s))
# consumer's purchase counts per (user, agent, seg)
upc = mdf.groupby(["user_id", "agent_id", "seg"]).size()

R = {}
n_cp = 0
v = {k: 0.0 for k in ["union_full", "both_full", "agentfull_pairwt", "jac", "overlap_min",
                      "containment", "inter_div8", "purch_jac", "purch_contain",
                      "purch_sharedseg_share", "inter_nonempty"]}
c1_list, c2_list, c3_list, c4_list = [], [], [], []

for u, alist in user_agents.items():
    m = len(alist)
    masks_u = [ag_mask[a] for a in alist]
    nseg_u = [ag_nseg[a] for a in alist]
    psets = [upseg.get((u, a), frozenset()) for a in alist]
    # per-consumer redundancy variants
    bought = {}
    for a in alist:
        for s in upseg.get((u, a), frozenset()):
            bought[s] = bought.get(s, 0) + 1
    tot_segs_bought = len(bought)
    segs_multi_bought = sum(1 for s, c in bought.items() if c > 1)
    c1_list.append(segs_multi_bought / tot_segs_bought if tot_segs_bought else 0.0)
    # purchase-weighted: purchases in segments bought from >=2 agents
    tot_purch = 0; red_purch = 0
    for a in alist:
        segc = upc.get((u, a))
        if segc is None:
            continue
        for s, c in segc.items():
            tot_purch += int(c)
            if bought.get(s, 0) > 1:
                red_purch += int(c)
    c2_list.append(red_purch / tot_purch if tot_purch else 0.0)
    # offered-by->=2 variants
    offered = {}
    for a in alist:
        for s in ag_seg[a]:
            offered[s] = offered.get(s, 0) + 1
    tot_off = len(offered)
    off_multi = sum(1 for s, c in offered.items() if c > 1)
    c4_list.append(off_multi / tot_off if tot_off else 0.0)
    # purchases in segments OFFERED by >=2 agents
    tot_p2 = 0; red_p2 = 0
    for a in alist:
        segc = upc.get((u, a))
        if segc is None:
            continue
        for s, c in segc.items():
            tot_p2 += int(c)
            if offered.get(s, 0) > 1:
                red_p2 += int(c)
    c3_list.append(red_p2 / tot_p2 if tot_p2 else 0.0)
    # consumer-pair variants
    for i in range(m):
        for j in range(i + 1, m):
            n_cp += 1
            mi, mj = masks_u[i], masks_u[j]
            inter = int(lut[mi & mj]); union = int(lut[mi | mj])
            v["union_full"] += 1.0 if union == n_seg else 0.0
            v["both_full"] += 1.0 if (mi == FULL and mj == FULL) else 0.0
            v["agentfull_pairwt"] += ((1.0 if mi == FULL else 0.0) + (1.0 if mj == FULL else 0.0)) / 2
            v["jac"] += inter / union if union else 1.0
            v["overlap_min"] += inter / min(nseg_u[i], nseg_u[j]) if min(nseg_u[i], nseg_u[j]) else 1.0
            v["containment"] += ((inter / nseg_u[i]) + (inter / nseg_u[j])) / 2
            v["inter_div8"] += inter / n_seg
            v["inter_nonempty"] += 1.0 if inter > 0 else 0.0
            Pi, Pj = psets[i], psets[j]
            pu_ = Pi | Pj; pi_ = Pi & Pj
            v["purch_jac"] += len(pi_) / len(pu_) if pu_ else 1.0
            if Pi and Pj:
                v["purch_contain"] += ((len(pi_) / len(Pi)) + (len(pi_) / len(Pj))) / 2
            # purchases in segments offered by both agents (this consumer's pair purchases)
            segs_both = set()
            for s in ag_seg[alist[i]]:
                if s in ag_seg[alist[j]]:
                    segs_both.add(s)
            tp = 0; rp = 0
            for a in (alist[i], alist[j]):
                segc = upc.get((u, a))
                if segc is None:
                    continue
                for s, c in segc.items():
                    tp += int(c)
                    if s in segs_both:
                        rp += int(c)
            v["purch_sharedseg_share"] += (rp / tp) if tp else 0.0

R["n_consumer_pairs"] = n_cp
for k in v:
    R[f"cpair_{k}"] = v[k] / n_cp
R["consumers_c1_distinctseg_bought_ge2agents_share"] = float(np.mean(c1_list))
R["consumers_c2_purchases_in_segs_bought_ge2agents_share"] = float(np.mean(c2_list))
R["consumers_c3_purchases_in_segs_offered_ge2agents_share"] = float(np.mean(c3_list))
R["consumers_c4_segs_offered_ge2_of_union_share"] = float(np.mean(c4_list))

with open(BASE + "vr28b_seg_variants_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
