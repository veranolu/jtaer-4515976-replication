# -*- coding: utf-8 -*-
"""
VR28-final item 8 (supplement): two verification tasks.
(A) focal LOPO support coverage with per-consumer definitions (v1 fidelity):
    per consumer: item-level share & purchase-weighted share of focal items with
    other-consumer support; report mean/median across 11,694 consumers.
    Targets: item mean 22.81% / median 14.29%; purchase mean 23.99% / median 14.29%.
(B) pop_orig benchmark re-test: weights = PLATFORM-WIDE global item popularity
    restricted to the pair's full shelf (incl. focal). Target: 24.939% / 0.0947.
    (agent-level counts incl. focal gave only 3.346% in vr28b_mc_lopo_20260906.)
seed=42, B=1000. Dual-path: vectorized lexsort overlap identical in structure to
vr28b_mc_lopo_20260906.py (already cross-checked vs targets on 3 benchmarks).
"""
import pandas as pd
import numpy as np
import json, time, datetime
from collections import Counter, defaultdict

t0 = time.time()
SCRIPT = "vr28b_poporig_support_20260906.py"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
B = 1000
SEED = 42
df = pd.read_csv(TRIP)
print(f"[{SCRIPT}] {datetime.datetime.now()} input triplets={len(df):,} B={B} seed={SEED}", flush=True)

six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
multi_users = six.loc[six["n_agents"] >= 2, "user_id"].to_numpy()

ai_cnt = df.groupby(["agent_id", "item_id"]).size()
ai_dict = ai_cnt.to_dict()
agent_shelf = df.groupby("agent_id")["item_id"].unique().apply(np.sort)
global_pop = df["item_id"].value_counts()

mdf = df[df["user_id"].isin(set(multi_users.tolist()))]
pair_items = mdf.groupby(["user_id", "agent_id"])["item_id"].agg(list)
pair_g = mdf.groupby(["user_id", "agent_id"]).size()

u_pairs = defaultdict(list)
for (u, a), lst in pair_items.items():
    u_pairs[u].append((a, lst, int(pair_g[(u, a)])))
consumers = sorted(u_pairs.keys())
R = {"n_consumers": len(consumers)}

# ---------- (A) per-consumer focal support coverage ----------
item_cov = np.zeros(len(consumers))
purch_cov = np.zeros(len(consumers))
for j, u in enumerate(consumers):
    it_in = it_tot = pu_in = pu_tot = 0
    for (a, lst, g) in u_pairs[u]:
        pc = Counter(lst)
        for it, c in pc.items():
            it_tot += 1
            pu_tot += c
            if ai_dict[(a, it)] - c > 0:
                it_in += 1
                pu_in += c
    item_cov[j] = it_in / it_tot if it_tot else np.nan
    purch_cov[j] = pu_in / pu_tot if pu_tot else np.nan
R["support_item_mean_pct"] = 100 * float(np.nanmean(item_cov))
R["support_item_median_pct"] = 100 * float(np.nanmedian(item_cov))
R["support_purch_mean_pct"] = 100 * float(np.nanmean(purch_cov))
R["support_purch_median_pct"] = 100 * float(np.nanmedian(purch_cov))
print(f"  (A) support item mean={R['support_item_mean_pct']:.4f}% med={R['support_item_median_pct']:.4f}% "
      f"| purch mean={R['support_purch_mean_pct']:.4f}% med={R['support_purch_median_pct']:.4f}% "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------- (B) pop_orig with global popularity weights ----------
# per-pair global-popularity-weighted shelves
pair_data = defaultdict(list)   # u -> list of (items, pv, g)
for (u, a), lst in pair_items.items():
    items = agent_shelf[a]
    fq = global_pop.reindex(items).to_numpy(dtype=np.float64)
    pv = fq / fq.sum()
    pair_data[u].append((items, pv, int(pair_g[(u, a)])))

rng = np.random.default_rng(SEED)
ov_all, rd_all = [], []
for u in consumers:
    recs = pair_data[u]
    draws = []
    for (items, pv, g) in recs:
        cw = np.cumsum(pv)
        idx = np.searchsorted(cw, rng.random((B, g)) * cw[-1])
        idx = np.minimum(idx, len(items) - 1)
        draws.append(items[idx])
    m = len(draws)
    rep_list, item_list, side_list = [], [], []
    base = np.arange(B)
    for si, d in enumerate(draws):
        g = d.shape[1]
        rep_list.append(np.repeat(base, g))
        item_list.append(d.ravel())
        side_list.append(np.full(B * g, si, dtype=np.int64))
    rep = np.concatenate(rep_list); itm = np.concatenate(item_list); side = np.concatenate(side_list)
    order = np.lexsort((side, itm, rep))
    rep_s, itm_s, side_s = rep[order], itm[order], side[order]
    new_grp = np.ones(len(rep_s), dtype=bool)
    new_grp[1:] = (rep_s[1:] != rep_s[:-1]) | (itm_s[1:] != itm_s[:-1])
    grp_id = np.cumsum(new_grp) - 1
    side_diff = np.ones(len(rep_s), dtype=np.int64)
    side_diff[1:] = ((side_s[1:] != side_s[:-1]) | new_grp[1:]).astype(np.int64)
    sides_per_grp = np.bincount(grp_id, weights=side_diff)
    rep_of_grp = rep_s[new_grp]
    multi = sides_per_grp >= 2
    distinct_per_rep = np.bincount(rep_of_grp, minlength=B)
    multi_per_rep = np.bincount(rep_of_grp[multi], minlength=B)
    ov_all.append(float((multi_per_rep > 0).mean()))
    with np.errstate(invalid="ignore", divide="ignore"):
        rd_all.append(float(np.where(distinct_per_rep > 0, multi_per_rep / distinct_per_rep, 0.0).mean()))

arr = np.array(ov_all); arrr = np.array(rd_all)
R["pop_orig_global_overlap_pct"] = 100 * float(arr.mean())
R["pop_orig_global_overlap_mcse_pct"] = 100 * float(arr.std(ddof=1) / np.sqrt(len(arr)))
R["pop_orig_global_redundancy"] = float(arrr.mean())
R["pop_orig_global_redundancy_mcse"] = float(arrr.std(ddof=1) / np.sqrt(len(arrr)))

R["targets"] = {
    "support_item_mean_pct": 22.81, "support_item_median_pct": 14.29,
    "support_purch_mean_pct": 23.99, "support_purch_median_pct": 14.29,
    "pop_orig_overlap_pct": 24.939, "pop_orig_redundancy": 0.0947}
with open(BASE + "vr28b_poporig_support_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
