# -*- coding: utf-8 -*-
"""
VR28-final items 3/4/7/10/12 + base table for 13.
Path A: fresh vectorized recomputation from raw triplets.
Path B: VR23 recalc npz (independent prior pipeline) cross-check to machine precision.
Log: script name, time, input rows. seed not needed (exact arithmetic).
"""
import pandas as pd
import numpy as np
from scipy.special import gammaln
from scipy import stats as sps
import json, time, datetime

t0 = time.time()
SCRIPT = "vr28b_pair_table_20260906.py"
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
NPZ = "/Coze/Drive/扣子/所有对话/主对话/VR23验核_20260904/recalc/"

df = pd.read_csv(TRIP)
n_in = len(df)
print(f"[{SCRIPT}] {datetime.datetime.now()} input triplets={n_in:,}", flush=True)

# ---------------- pair table ----------------
g = df.groupby(["user_id", "agent_id"])
pair = g.size().rename("g").reset_index()
pair["d"] = g["item_id"].nunique().values
# H_obs
cnt = df.groupby(["user_id", "agent_id", "item_id"]).size().rename("c").reset_index()
cnt["p"] = cnt["c"] / cnt.groupby(["user_id", "agent_id"])["c"].transform("sum")
H = cnt.groupby(["user_id", "agent_id"]).apply(lambda x: float(-(x["p"] * np.log(x["p"])).sum()), include_groups=False)
pair["H_obs"] = pair.set_index(["user_id", "agent_id"]).index.map(H).astype(float)
# shelf size per agent
shelf = df.groupby("agent_id")["item_id"].nunique()
pair["n_s"] = pair["agent_id"].map(shelf).astype(int)
pair = pair.sort_values(["user_id", "agent_id"]).reset_index(drop=True)
N_pairs = len(pair)
assert N_pairs == 101577

pu = pair["user_id"].to_numpy(); pa = pair["agent_id"].to_numpy()
pg = pair["g"].to_numpy(); pns = pair["n_s"].to_numpy()
pd_ = pair["d"].to_numpy(); pH = pair["H_obs"].to_numpy()

# ---------------- E[H|g,p] machinery: f(g,p)=E[-(X/g)ln(X/g)], X~Bin(g,p) ----------------
def f_binom(g, p):
    """vector over p for scalar g"""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    pos = p > 0
    if not pos.any():
        return out
    pv = p[pos]
    lp = np.log(pv); lq = np.log1p(-pv)
    acc = np.zeros_like(pv)
    ks = np.arange(1, g + 1, dtype=np.float64)
    t = (ks / g) * np.log(ks / g)  # h(k/g)*(-1) factor handled outside
    logC = gammaln(g + 1) - gammaln(ks + 1) - gammaln(g - ks + 1)
    for j in range(len(ks)):
        acc += np.exp(logC[j] + ks[j] * lp + (g - ks[j]) * lq) * t[j]
    out[pos] = -acc
    return out

# ---------------- E_u (uniform), cached on (g,n_s) ----------------
cache_u = {}
def eu_scalar(g, n):
    key = (int(g), int(n))
    v = cache_u.get(key)
    if v is None:
        if g <= 1 or n <= 1:
            v = 0.0
        else:
            v = float(n) * float(f_binom(g, np.array([1.0 / n]))[0])
        cache_u[key] = v
    return v

uniq_gn = np.unique(np.stack([pg, pns], axis=1), axis=0)
for gv, nv in uniq_gn:
    eu_scalar(gv, nv)
E_u = np.array([cache_u[(int(a), int(b))] for a, b in zip(pg, pns)])

# ---------------- shelf membership long table for E_g / E_s ----------------
# agent -> shelf items, agent-item counts
agent_items = df.groupby("agent_id")["item_id"].apply(lambda s: s.to_numpy())
ai_cnt = df.groupby(["agent_id", "item_id"]).size()
# per-pair item counts dict for LOPO subtraction
pair_cnt = cnt.set_index(["user_id", "agent_id", "item_id"])["c"]

rows_pi, rows_item, rows_wfull, rows_wloo = [], [], [], []
item_pop = df["item_id"].value_counts()
for i in range(N_pairs):
    u, a = pu[i], pa[i]
    items = np.unique(agent_items[a])
    ac = ai_cnt.loc[a].reindex(items).to_numpy(dtype=np.float64)
    # pair's own counts on shelf items
    pc = np.array([pair_cnt.get((u, a, it), 0) for it in items], dtype=np.float64)
    rows_pi.append(np.full(len(items), i, dtype=np.int64))
    rows_item.append(items)
    rows_wfull.append(ac)
    rows_wloo.append(ac - pc)
    if (i + 1) % 30000 == 0:
        print(f"  shelf long table {i+1:,}/{N_pairs:,} ({time.time()-t0:.0f}s)", flush=True)

pi_idx = np.concatenate(rows_pi)
lt_item = np.concatenate(rows_item)
w_full = np.concatenate(rows_wfull)
w_loo = np.concatenate(rows_wloo)
print(f"  long table rows={len(pi_idx):,} ({time.time()-t0:.0f}s)", flush=True)

pg_row = pg[pi_idx]
glob = item_pop.reindex(lt_item).to_numpy(dtype=np.float64)

def bench_E(w):
    """E[H|g,p] per pair with weights w over shelf items"""
    s = pd.Series(w).groupby(pi_idx).sum().to_numpy()
    p = w / s[pi_idx]
    pos = w > 0
    contrib = np.zeros(len(w))
    # process per distinct g to keep k-loop cheap
    for gv in np.unique(pg_row):
        m = (pg_row == gv) & pos
        if gv <= 1 or not m.any():
            continue
        contrib[m] = f_binom(int(gv), p[m])
    E = pd.Series(contrib).groupby(pi_idx).sum().to_numpy()
    # pairs whose total weight <=0 -> fallback flag
    fb = s <= 0
    return E, fb, s

E_g, fb_g, s_full = bench_E(w_full)          # pop_global uses full weights? NO --
# pop_global: p = global popularity normalized within shelf
s_glob = pd.Series(glob).groupby(pi_idx).sum().to_numpy()
p_glob = glob / s_glob[pi_idx]
contrib = np.zeros(len(glob))
for gv in np.unique(pg_row):
    m = pg_row == gv
    if gv <= 1 or not m.any():
        continue
    contrib[m] = f_binom(int(gv), p_glob[m])
E_g = pd.Series(contrib).groupby(pi_idx).sum().to_numpy()

# pop_shelf LOPO
E_s, fb_s, s_loo = None, None, None
w = w_loo
s = pd.Series(w).groupby(pi_idx).sum().to_numpy()
p = np.divide(w, s[pi_idx], out=np.zeros_like(w), where=s[pi_idx] > 0)
pos = w > 0
contrib = np.zeros(len(w))
for gv in np.unique(pg_row):
    m = (pg_row == gv) & pos
    if gv <= 1 or not m.any():
        continue
    contrib[m] = f_binom(int(gv), p[m])
E_s_raw = pd.Series(contrib).groupby(pi_idx).sum().to_numpy()
fb_s = s <= 0
# fallback to uniform if LOPO shelf empty
E_s = np.where(fb_s, E_u, E_s_raw)
print(f"  benchmarks done ({time.time()-t0:.0f}s) fallback LOPO={int(fb_s.sum())}", flush=True)

# ---------------- coverage expectation ----------------
E_uniq = pns * (1.0 - ((pns - 1.0) / pns) ** pg)

# ---------------- Path B cross-check vs VR23 npz ----------------
d1 = np.load(NPZ + "recalc_stage1.npz"); d2 = np.load(NPZ + "recalc_stage2.npz"); dc = np.load(NPZ + "recalc_coverage.npz")
ug_b = d1["ug"]
keyA = pd.MultiIndex.from_arrays([pu, pa])
keyB = pd.MultiIndex.from_arrays([ug_b[:, 0], ug_b[:, 1]])
orderB = keyB.get_indexer(keyA)
assert (orderB >= 0).all()
def maxdiff(x, y):
    return float(np.max(np.abs(x - y[orderB])))
xcheck = {
    "g": maxdiff(pg.astype(float), d1["pair_g"].astype(float)),
    "n_s": maxdiff(pns.astype(float), d1["pair_ns"].astype(float)),
    "H_obs": maxdiff(pH, d1["H_obs"]),
    "E_u": maxdiff(E_u, d1["E_u"]),
    "E_g": maxdiff(E_g, d1["E_g"]),
    "E_s": maxdiff(E_s, d2["E_s"]),
    "pair_nd": maxdiff(pd_.astype(float), dc["pair_nd"]),
    "E_uniq": maxdiff(E_uniq, dc["E_uniq"]),
    "loo_fb": float(np.max(np.abs(fb_s.astype(float) - d2["loo_fb"][orderB].astype(float)))),
}
print("xcheck maxdiff vs npz:", json.dumps(xcheck, indent=1), flush=True)
assert max(v for k, v in xcheck.items() if k != "loo_fb") < 1e-9
assert xcheck["loo_fb"] == 0.0

# ---------------- save pair-level detail CSV ----------------
out_pair = BASE + "vr28b_pair_level_101577_20260906.csv"
pair_out = pair.copy()
pair_out["E_u"] = E_u; pair_out["E_s"] = E_s; pair_out["E_g"] = E_g; pair_out["E_uniq"] = E_uniq
pair_out.to_csv(out_pair, index=False)
print(f"  pair CSV saved: {out_pair} rows={len(pair_out):,}", flush=True)

# ---------------- stats ----------------
R = {}
g2 = pg >= 2
R["n_pairs_g2"] = int(g2.sum())

# item 7: benchmark means over g>=2 + below shares
R["mean_Hobs_g2"] = float(pH[g2].mean())
R["mean_Eu_g2"] = float(E_u[g2].mean())
R["mean_Es_g2"] = float(E_s[g2].mean())
R["mean_Eg_g2"] = float(E_g[g2].mean())
R["below_uniform_pct"] = float(100 * (pH[g2] < E_u[g2]).mean())
R["below_popshelf_pct"] = float(100 * (pH[g2] < E_s[g2]).mean())
R["below_popglobal_pct"] = float(100 * (pH[g2] < E_g[g2]).mean())

# ratio stats with masks
mask_u = g2 & (E_u > 0)
mask_s = g2 & (E_s > 0)
mask_g = g2 & (E_g > 0)
R["N_Ru"] = int(mask_u.sum()); R["N_Rs"] = int(mask_s.sum()); R["N_Rg"] = int(mask_g.sum())
Ru = pH / np.where(E_u > 0, E_u, np.nan)
Rs = pH / np.where(E_s > 0, E_s, np.nan)
Rg = pH / np.where(E_g > 0, E_g, np.nan)
for nm, v, m in [("Ru", Ru, mask_u), ("Rs", Rs, mask_s), ("Rg", Rg, mask_g)]:
    R[f"{nm}_mean"] = float(np.nanmean(v[m]))
    R[f"{nm}_median"] = float(np.nanmedian(v[m]))
R["RoM_u"] = float(pH[g2].mean() / E_u[g2].mean())
R["RoM_s"] = float(pH[g2].mean() / E_s[g2].mean())
R["RoM_g"] = float(pH[g2].mean() / E_g[g2].mean())
J = pH[g2] / np.log(pg[g2])
R["Jg_mean"] = float(J.mean()); R["Jg_median"] = float(np.median(J))

# item 4: masks 177/215/181/34
mask177 = g2 & (pns == 1)
R["mask_uniform_ns1"] = int(mask177.sum())
R["mask_popshelf_E0"] = int((g2 & (E_s_raw <= 0)).sum())
R["mask_s_and_Hobs0"] = int((g2 & (E_s_raw <= 0) & (pH == 0)).sum())
R["mask_s_and_Hobs_pos"] = int((g2 & (E_s_raw <= 0) & (pH > 0)).sum())
R["mask177_subsetof_215"] = bool(((g2 & (E_s_raw <= 0)) | ~mask177).all())
R["mask177_Hobs0"] = int((mask177 & (pH == 0)).sum())
R["extra215_vs_177"] = int(((g2 & (E_s_raw <= 0)) & ~mask177).sum())
R["extra215_Hobs0"] = int(((g2 & (E_s_raw <= 0)) & ~mask177 & (pH == 0)).sum())
R["extra215_Hobs_pos"] = int(((g2 & (E_s_raw <= 0)) & ~mask177 & (pH > 0)).sum())
# E_u==0 equivalent check
R["mask_Eu0_eq_ns1"] = bool((g2 & (E_u <= 0)).sum() == int(mask177.sum()))
R["N_Rs_check_86710_minus_215"] = int(g2.sum()) - int((g2 & (E_s_raw <= 0)).sum())

# item 3: static ratio H_obs/ln(n_s)
stat_ratio = pH[g2] / np.log(pns[g2])
R["static_mean_overall"] = float(stat_ratio.mean())
R["static_median_overall"] = float(np.median(stat_ratio))
R["static_RoM_overall"] = float(pH[g2].mean() / np.log(pns[g2]).mean())
tiers = [(1, 50), (51, 200), (201, 272)]
for lo, hi in tiers:
    m = g2 & (E_u > 0) & (pns >= lo) & (pns <= hi)
    mt = g2 & (pns >= lo) & (pns <= hi)
    R[f"tier_{lo}_{hi}_N_masked"] = int(m.sum())
    R[f"tier_{lo}_{hi}_N_all"] = int(mt.sum())
    R[f"tier_{lo}_{hi}_mean_obs"] = float(pH[m].mean())
    R[f"tier_{lo}_{hi}_mean_exp"] = float(E_u[m].mean())
    R[f"tier_{lo}_{hi}_mean_ratio"] = float(np.nanmean(Ru[m]))
    R[f"tier_{lo}_{hi}_median_ratio"] = float(np.nanmedian(Ru[m]))
    R[f"tier_{lo}_{hi}_static_mean"] = float((pH[mt] / np.log(pns[mt])).mean())
# RQ2 r=0.41
R["rq2_pearson_ns_Hobs_g2"] = float(sps.pearsonr(pns[g2], pH[g2])[0])
R["rq2_pearson_ns_Hobs_all"] = float(sps.pearsonr(pns, pH)[0])
R["rq2_spearman_ns_Hobs_g2"] = float(sps.spearmanr(pns[g2], pH[g2])[0])

# coverage
R["cov_obs_mean_full"] = float(pd_.mean())
R["cov_exp_mean_full"] = float(E_uniq.mean())
R["cov_RoM_full"] = float(pd_.mean() / E_uniq.mean())
R["cov_RoM_g2"] = float(pd_[g2].mean() / E_uniq[g2].mean())
R["cov_mean_of_ratios_full"] = float((pd_ / E_uniq).mean())

# item 10: g=5,S=10 worked example
e5_10 = eu_scalar(5, 10)
R["worked_g5_S10"] = e5_10
R["worked_ln10"] = float(np.log(10))
R["worked_share_pct"] = float(100 * e5_10 / np.log(10))

# item 12: assorted anchors restated from pair table
R["pairs_total"] = int(N_pairs)
R["pairs_g1"] = int((pg == 1).sum())
R["pairs_g1_pct"] = float(100 * (pg == 1).mean())

with open(BASE + "vr28b_pair_results_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
