# -*- coding: utf-8 -*-
"""
VR28-final item 3 (static ratio variants) + item 1 (purchase count distribution).
Input: vr28b_pair_level_101577_20260906.csv + six-var user table.
"""
import pandas as pd
import numpy as np
import json, time, datetime

t0 = time.time()
SCRIPT = "vr28b_static_item1_20260906.py"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
print(f"[{SCRIPT}] {datetime.datetime.now()}", flush=True)

pair = pd.read_csv(BASE + "vr28b_pair_level_101577_20260906.csv")
n_in = len(pair)
print(f"input pairs={n_in:,}", flush=True)
pg = pair["g"].to_numpy(); pns = pair["n_s"].to_numpy()
pH = pair["H_obs"].to_numpy(); Eu = pair["E_u"].to_numpy()

g2 = pg >= 2
R = {}
# static ratio variants (ln(n_s) undefined at n_s==1 -> 177 pairs excluded or zero-filled)
sr = np.where(pns > 1, pH / np.log(np.maximum(pns, 2)), np.nan)
m_g2_ns1 = g2 & (pns == 1)
R["g2_ns1_pairs"] = int(m_g2_ns1.sum())
variants = {}
sel = g2 & (pns > 1)
variants["mean_g2_excl_ns1"] = float(np.nanmean(sr[sel]))
variants["median_g2_excl_ns1"] = float(np.nanmedian(sr[sel]))
variants["RoM_g2_excl_ns1"] = float(pH[sel].mean() / np.log(pns[sel]).mean())
zf = np.where(pns > 1, pH / np.log(np.maximum(pns, 2)), 0.0)
variants["mean_g2_zerofill"] = float(zf[g2].mean())
variants["median_g2_zerofill"] = float(np.median(zf[g2]))
selEu = g2 & (Eu > 0)
variants["mean_g2_EuPos"] = float(np.nanmean(sr[selEu]))
variants["median_g2_EuPos"] = float(np.nanmedian(sr[selEu]))
variants["RoM_g2_EuPos"] = float(pH[selEu].mean() / np.log(pns[selEu]).mean())
R["variants"] = variants
for lo, hi in [(1, 50), (51, 200), (201, 272)]:
    m = g2 & (pns >= lo) & (pns <= hi) & (pns > 1)
    R[f"tier{lo}_{hi}_static_mean"] = float(np.nanmean(sr[m]))
    R[f"tier{lo}_{hi}_static_median"] = float(np.nanmedian(sr[m]))
    R[f"tier{lo}_{hi}_static_RoM"] = float(pH[m].mean() / np.log(pns[m]).mean())
    R[f"tier{lo}_{hi}_N_excl_ns1"] = int(m.sum())

# item 1: purchase count distribution
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
npur = six["n_purchases"].to_numpy()
N = len(npur)
srt = np.sort(npur)
R["N_users"] = N
R["median_position_value"] = int(srt[(N + 1) // 2 - 1])  # 43553rd (1-based)
R["median_np"] = float(np.median(npur))
vc = np.bincount(npur)
le5 = int(vc[:6].sum())
R["le5_count"] = le5
R["le5_pct"] = 100.0 * le5 / N
R["le5_pct_4dp"] = round(100.0 * le5 / N, 4)
dist = []
cum = 0
for k in range(1, 11):
    c = int(vc[k]) if k < len(vc) else 0
    cum += c
    dist.append({"purchases": k, "users": c, "cum_pct": round(100.0 * cum / N, 4)})
R["dist_top10"] = dist
# dual path: value_counts path
vc2 = six["n_purchases"].value_counts().sort_index()
assert int(vc2[vc2.index <= 5].sum()) == le5
assert int(vc2.iloc[:10].sum()) == cum

with open(BASE + "vr28b_static_item1_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
print(f"[{SCRIPT}] DONE ({time.time()-t0:.0f}s)", flush=True)
