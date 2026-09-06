# -*- coding: utf-8 -*-
"""
VR28-final item 3 round 2: static ratio variant matrix incl. g=1 pairs.
"""
import pandas as pd
import numpy as np
import json, time, datetime

t0 = time.time()
SCRIPT = "vr28b_static_v2_20260906.py"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
print(f"[{SCRIPT}] {datetime.datetime.now()}", flush=True)
pair = pd.read_csv(BASE + "vr28b_pair_level_101577_20260906.csv")
print(f"input pairs={len(pair):,}", flush=True)
pg = pair["g"].to_numpy(); pns = pair["n_s"].to_numpy(); pH = pair["H_obs"].to_numpy()

R = {}
sr = np.where(pns > 1, pH / np.log(np.maximum(pns, 2)), np.nan)
zf = np.where(pns > 1, pH / np.log(np.maximum(pns, 2)), 0.0)
lnns = np.log(np.maximum(pns, 2))

scopes = {
    "all_pairs": np.ones(len(pair), bool),
    "g2": pg >= 2,
}
for sc, m in scopes.items():
    R[f"{sc}_mean_zerofill"] = float(np.nanmean(zf[m]))
    R[f"{sc}_median_zerofill"] = float(np.median(zf[m]))
    R[f"{sc}_RoM_zerofill"] = float(pH[m].mean() / lnns[m].mean())
    mm = m & (pns > 1)
    R[f"{sc}_mean_excl"] = float(np.nanmean(sr[mm]))
    R[f"{sc}_RoM_excl"] = float(pH[mm].mean() / lnns[mm].mean())

for lo, hi in [(1, 50), (51, 200), (201, 272)]:
    m = (pns >= lo) & (pns <= hi)
    R[f"tier{lo}_{hi}_allN"] = int(m.sum())
    R[f"tier{lo}_{hi}_all_mean_zf"] = float(np.nanmean(zf[m]))
    R[f"tier{lo}_{hi}_all_RoM"] = float(pH[m].mean() / lnns[m].mean())
    m2 = m & (pg >= 2)
    R[f"tier{lo}_{hi}_g2_mean_zf"] = float(np.nanmean(zf[m2]))
    R[f"tier{lo}_{hi}_g2_RoM"] = float(pH[m2].mean() / lnns[m2].mean())

# context: purchases of multi-agent consumers / single-agent mean g
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
npur = six["n_purchases"].to_numpy(); nag = six["n_agents"].to_numpy()
R["mean_g_single_agent_consumers"] = float(npur[nag == 1].mean())
R["mean_g_multi_consumers"] = float(npur[nag >= 2].mean())
R["total_purchases_multi"] = int(npur[nag >= 2].sum())

# consumer-level static ratio (single-agent consumers: H of their only pair / ln shelf)
fig2 = pd.read_csv(BASE + "fig2_user_hhi_topshare_87105_20260906.csv")
R["mean_ln_ns_pairs"] = float(lnns.mean())
R["mean_H_all_pairs"] = float(pH.mean())

with open(BASE + "vr28b_static_v2_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
