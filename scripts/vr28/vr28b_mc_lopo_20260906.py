# -*- coding: utf-8 -*-
"""
VR28-final item 8: Monte Carlo null benchmarks for cross-agent overlap.
Benchmarks: expected non-zero overlap & redundancy, uniform/popularity x original/LOPO.
B=1000, seed=42 (task-wide). Targets: LOPO 2.095%/3.194% (red 0.0028/0.0056);
original 2.142%/24.939% (red 0.0026/0.0947). RNG layout is ours; compare within MC error.
Also exact S10 extras: fallback count, shelf-size ratio, focal-item support stats.
"""
import pandas as pd
import numpy as np
import json, time, datetime

t0 = time.time()
SCRIPT = "vr28b_mc_lopo_20260906.py"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
B = 1000
SEED = 42
df = pd.read_csv(TRIP)
print(f"[{SCRIPT}] {datetime.datetime.now()} input triplets={len(df):,} B={B} seed={SEED}", flush=True)

six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
multi_users = set(six.loc[six["n_agents"] >= 2, "user_id"].tolist())

# ---------- per-pair shelves ----------
agent_item_cnt = df.groupby(["agent_id", "item_id"]).size()
agent_shelf_items = df.groupby("agent_id")["item_id"].unique().apply(np.sort)
pair_cnt = df.groupby(["user_id", "agent_id", "item_id"]).size()
pair_g = df.groupby(["user_id", "agent_id"]).size()

mdf = df[df["user_id"].isin(multi_users)]
pairs = list(mdf.groupby(["user_id", "agent_id"]).size().index)
R = {"n_pairs_multi": len(pairs)}

# build per-pair arrays
pair_data = []
empty_lopo = 0
shelf_ratio = []
support_share_pair = []   # per pair: share of distinct items with other-consumer support
supp_purch = 0; tot_purch = 0
for (u, a) in pairs:
    items = agent_shelf_items[a]
    ac = agent_item_cnt.loc[a].reindex(items).to_numpy(dtype=np.float64)
    pc = np.array([pair_cnt.get((u, a, it), 0) for it in items], dtype=np.float64)
    loo = ac - pc
    pos = loo > 0
    if pos.sum() == 0:
        empty_lopo += 1
    li = items[pos]; lw = loo[pos]
    shelf_ratio.append(pos.sum() / len(items))
    # support stats: pair's own purchased items having other-consumer support
    own_items = items[pc > 0]
    own_cnts = pc[pc > 0]
    supp = (ac[pc > 0] - own_cnts) > 0
    support_share_pair.append(float(supp.mean()))
    supp_purch += int((own_cnts[supp]).sum()); tot_purch += int(own_cnts.sum())
    pair_data.append((u, a, li, lw, items, ac, int(pair_g[(u, a)])))

R["lopo_empty_fallback"] = empty_lopo
R["shelf_ratio_mean"] = float(np.mean(shelf_ratio))
R["shelf_ratio_median"] = float(np.median(shelf_ratio))
R["support_share_mean_pct"] = 100 * float(np.mean(support_share_pair))
R["support_share_median_pct"] = 100 * float(np.median(support_share_pair))
R["support_purchase_weighted_pct"] = 100 * supp_purch / tot_purch
print(f"  pair data built ({time.time()-t0:.0f}s) empty_lopo={empty_lopo}", flush=True)

# group pairs by consumer
from collections import defaultdict
cons_pairs = defaultdict(list)
for rec in pair_data:
    cons_pairs[rec[0]].append(rec)
consumers = sorted(cons_pairs.keys())
R["n_consumers"] = len(consumers)

rng = np.random.default_rng(SEED)

def mc_consumer(recs, mode):
    """mode in {uni_lopo, pop_lopo, uni_orig, pop_orig}. Returns (overlap_mean, red_mean) over B."""
    draws = []
    for (u, a, li, lw, oi, ow, g) in recs:
        if mode == "uni_lopo":
            items, w = li, lw
            idx = rng.integers(0, len(items), size=(B, g))
        elif mode == "pop_lopo":
            items, w = li, lw
            cw = np.cumsum(w)
            idx = np.searchsorted(cw, rng.random((B, g)) * cw[-1])
            idx = np.minimum(idx, len(items) - 1)
        elif mode == "uni_orig":
            items, w = oi, ow
            idx = rng.integers(0, len(items), size=(B, g))
        else:
            items, w = oi, ow
            cw = np.cumsum(w)
            idx = np.searchsorted(cw, rng.random((B, g)) * cw[-1])
            idx = np.minimum(idx, len(items) - 1)
        draws.append(items[idx])  # (B, g) item ids
    # encode elements: (rep, item, side)
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
    # group boundaries by (rep, item)
    new_grp = np.ones(len(rep_s), dtype=bool)
    new_grp[1:] = (rep_s[1:] != rep_s[:-1]) | (itm_s[1:] != itm_s[:-1])
    grp_id = np.cumsum(new_grp) - 1
    # distinct sides per group (sides sorted within group due to lexsort)
    side_diff = np.ones(len(rep_s), dtype=np.int64)
    side_diff[1:] = ((side_s[1:] != side_s[:-1]) | new_grp[1:]).astype(np.int64)
    sides_per_grp = np.bincount(grp_id, weights=side_diff)
    rep_of_grp = rep_s[new_grp]
    multi = sides_per_grp >= 2
    # per replicate stats
    nrep = B
    distinct_per_rep = np.bincount(rep_of_grp, minlength=nrep)
    multi_per_rep = np.bincount(rep_of_grp[multi], minlength=nrep)
    overlap = (multi_per_rep > 0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(distinct_per_rep > 0, multi_per_rep / distinct_per_rep, 0.0)
    return float(overlap.mean()), float(rate.mean())

modes = ["uni_lopo", "pop_lopo", "uni_orig", "pop_orig"]
acc = {m: [] for m in modes}
accr = {m: [] for m in modes}
tlog = time.time()
for ci, u in enumerate(consumers):
    recs = cons_pairs[u]
    for m in modes:
        ov, rd = mc_consumer(recs, m)
        acc[m].append(ov); accr[m].append(rd)
    if (ci + 1) % 2000 == 0:
        print(f"  MC {ci+1:,}/{len(consumers):,} ({time.time()-tlog:.0f}s)", flush=True)

for m in modes:
    arr = np.array(acc[m]); arrr = np.array(accr[m])
    R[f"{m}_overlap_pct"] = 100 * float(arr.mean())
    R[f"{m}_overlap_mcse_pct"] = 100 * float(arr.std(ddof=1) / np.sqrt(len(arr)))
    R[f"{m}_redundancy"] = float(arrr.mean())
    R[f"{m}_redundancy_mcse"] = float(arrr.std(ddof=1) / np.sqrt(len(arrr)))

with open(BASE + "vr28b_mc_lopo_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
