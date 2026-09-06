# -*- coding: utf-8 -*-
"""
VR28-final item 13: B=10,000 consumer (cluster) bootstrap, seed=42, full-text CI audit.
13 statistics: 4 ratio-of-means (coverage full/g2, uniform, pop_shelf, pop_global),
3 mean-of-ratios (R_u, R_s, R_g), 3 Table-5 shelf-tier MoRs, HHI mean/median (multi-agent).
Percentile CI at B=1,000 (first 1,000 reps of the same stream) and B=10,000, plus BCa
(closed-form jackknife LOO acceleration) at B=10,000.
RNG: np.random.RandomState(42), per-rep randint(0, n_users, n_users) — mirrors
recalc_B.py stream so the B=1,000 R_u CI should EXACTLY reproduce the R3-letter
anchor [0.9792, 0.9822]. Special check printed for 0.9807.
Runtime recorded. Dual-path discipline: point estimates re-derived here from the
pair-level CSV must equal vr28b_pair_results_20260906.json anchors before bootstrap.
"""
import pandas as pd
import numpy as np
import json, time, datetime
from scipy.stats import norm

t0 = time.time()
SCRIPT = "vr28b_bootstrap_20260906.py"
BASE = "/Coze/Drive/扣子/所有对话/主对话/VR24输出_20260904/"
SEED = 42
B = 10000

pairs = pd.read_csv(BASE + "vr28b_pair_level_101577_20260906.csv")
six = pd.read_csv(BASE + "user_level_6vars_full686k_20260905.csv")
print(f"[{SCRIPT}] {datetime.datetime.now()} pairs={len(pairs):,} users={len(six):,} B={B} seed={SEED}", flush=True)

users = np.sort(six["user_id"].unique())
n_users = len(users)
uid_map = np.zeros(users.max() + 1, dtype=np.int64) - 1
uid_map[users] = np.arange(n_users)
puidx = uid_map[pairs["user_id"].to_numpy()]

g = pairs["g"].to_numpy()
H = pairs["H_obs"].to_numpy()
ns = pairs["n_s"].to_numpy()
d = pairs["d"].to_numpy()
Eu = pairs["E_u"].to_numpy()
Es = pairs["E_s"].to_numpy()
Eg = pairs["E_g"].to_numpy()
Euniq = pairs["E_uniq"].to_numpy()
g2 = g >= 2
mask_u = g2 & (ns >= 2)          # R_u / R_g / tiers universe (86,533)
mask_s = g2 & (Es > 0)           # R_s universe (86,495)
tiers = [(1, 50), (51, 200), (201, 272)]

def agg(values, weights):
    """per-consumer sums of weights (length n_pairs) restricted by boolean values-> handled outside"""
    out = np.zeros(n_users)
    np.add.at(out, puidx, weights)
    return out

SPECS = []  # (name, num_agg, den_agg, N_pairs, point, paper_ci_text, paper_lo, paper_hi, universe)
def add_rom(name, num, den, mask, paper_ci, plo, phi, universe):
    num_a = agg(None, np.where(mask, num, 0.0))
    den_a = agg(None, np.where(mask, den, 0.0))
    point = num_a.sum() / den_a.sum()
    SPECS.append(dict(name=name, num=num_a, den=den_a, N=int(mask.sum()), point=float(point),
                      paper_ci=paper_ci, plo=plo, phi=phi, universe=universe))
def add_mor(name, ratio, mask, paper_ci, plo, phi, universe):
    num_a = agg(None, np.where(mask, ratio, 0.0))
    den_a = agg(None, mask.astype(np.float64))
    point = num_a.sum() / den_a.sum()
    SPECS.append(dict(name=name, num=num_a, den=den_a, N=int(mask.sum()), point=float(point),
                      paper_ci=paper_ci, plo=plo, phi=phi, universe=universe))

add_rom("coverage_RoM_full", d, Euniq, np.ones(len(pairs), bool), "[1.027,1.030]", 1.027, 1.030, "all 101,577 pairs, cluster 87,105 consumers")
add_rom("coverage_RoM_g2", d, Euniq, g2, "(no paper CI; recalc ref)", np.nan, np.nan, "86,710 g>=2 pairs")
add_rom("uniform_RoM_g2", H, Eu, g2, "[0.993,0.995]", 0.993, 0.995, "86,710 g>=2 pairs")
add_rom("popshelf_RoM_g2", H, Es, g2, "[1.028,1.031]", 1.028, 1.031, "86,710 g>=2 pairs (E_s=0 kept in sums)")
add_rom("popglobal_RoM_g2", H, Eg, g2, "[1.819,1.832]", 1.819, 1.832, "86,710 g>=2 pairs")
add_mor("R_u_MoR", H / np.where(Eu > 0, Eu, np.nan), mask_u, "[0.979,0.982] main / [0.9792,0.9822] R3-letter", 0.9792, 0.9822, "86,533 pairs g>=2 & n_s>=2")
add_mor("R_s_MoR", H / np.where(Es > 0, Es, np.nan), mask_s, "[1.015,1.019] main / [1.0148,1.0189] R3-letter", 1.0148, 1.0189, "86,495 pairs g>=2 & E_s>0")
add_mor("R_g_MoR", H / np.where(Eg > 0, Eg, np.nan), mask_u, "[2.4160,2.4539]", 2.4160, 2.4539, "86,533 pairs g>=2 & n_s>=2")
ratio_u = H / np.where(Eu > 0, Eu, np.nan)
for (lo, hi), pc, plo, phi in zip(tiers, ["[0.965,0.970]", "[0.998,1.001]", "[0.980,0.996]"], [0.965, 0.998, 0.980], [0.970, 1.001, 0.996]):
    mt = g2 & (ns >= lo) & (ns <= hi) & (ns >= 2)
    add_mor(f"Table5_tier_{lo}_{hi}", ratio_u, mt, pc, plo, phi, f"g>=2 & n_s in [{lo},{hi}] & n_s>=2")

# ---- point-estimate fidelity gate vs vr28b_pair_results anchors ----
ANCH = {"coverage_RoM_full": 1.0289492218225114, "coverage_RoM_g2": 1.0296682974519455,
        "uniform_RoM_g2": 0.9940848304288114, "popshelf_RoM_g2": 1.029351469042679,
        "popglobal_RoM_g2": 1.8254654836209114, "R_u_MoR": 0.9806639415186835,
        "R_s_MoR": 1.0169298245009357, "R_g_MoR": 2.4336594554695212,
        "Table5_tier_1_50": 0.9677286597424588, "Table5_tier_51_200": 0.9992994393778326,
        "Table5_tier_201_272": 0.9884917564406691}
for s in SPECS:
    a = ANCH[s["name"]]
    assert abs(s["point"] - a) < 1e-12, f"{s['name']} point {s['point']} != anchor {a}"
print(f"  point-estimate gate: 11/11 pair stats == pair_results anchors ({time.time()-t0:.0f}s)", flush=True)

# ---- HHI (multi-agent consumers, iid over 11,694) ----
hhi_multi = six.loc[six["n_agents"] >= 2, "HHI"].to_numpy()
n_hhi = len(hhi_multi)
hhi_mean_point = float(hhi_multi.mean()); hhi_med_point = float(np.median(hhi_multi))
print(f"  HHI multi N={n_hhi:,} mean={hhi_mean_point:.6f} (锚0.6050) median={hhi_med_point:.6f} (锚0.5634)", flush=True)

# ---- bootstrap loop (single shared stream, RandomState(42), mirrors recalc_B) ----
rng = np.random.RandomState(SEED)
boot = np.empty((B, len(SPECS)))
tb = time.time()
for b in range(B):
    su = rng.randint(0, n_users, size=n_users)
    cnt = np.bincount(su, minlength=n_users).astype(np.float64)
    for j, s in enumerate(SPECS):
        boot[b, j] = (cnt @ s["num"]) / (cnt @ s["den"])
print(f"  pair-stats bootstrap done ({time.time()-tb:.0f}s)", flush=True)

rng_h = np.random.RandomState(SEED)
boot_hhi_mean = np.empty(B); boot_hhi_med = np.empty(B)
tb = time.time()
for b in range(B):
    su = rng_h.randint(0, n_hhi, size=n_hhi)
    x = hhi_multi[su]
    boot_hhi_mean[b] = x.mean()
    boot_hhi_med[b] = np.median(x)
print(f"  HHI bootstrap done ({time.time()-tb:.0f}s)", flush=True)

# ---- jackknife LOO (closed form) + BCa ----
def bca(boots, loo, point):
    z0 = norm.ppf(np.clip(np.mean(boots < point), 1e-12, 1 - 1e-12))
    ml = loo.mean()
    dd = ml - loo
    acc = (dd ** 3).sum() / (6.0 * (dd ** 2).sum() ** 1.5)
    out = []
    for alpha in (0.025, 0.975):
        za = norm.ppf(alpha)
        a1 = norm.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za)))
        out.append(float(np.percentile(boots, 100 * a1)))
    return out, float(z0), float(acc)

rows = []
for j, s in enumerate(SPECS):
    SN, SD = s["num"].sum(), s["den"].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (SN - s["num"]) / (SD - s["den"])
    loo = np.where(s["den"] > 0, loo, s["point"])
    b10 = boot[:, j]
    pct1k = np.percentile(b10[:1000], [2.5, 97.5])
    pct10k = np.percentile(b10, [2.5, 97.5])
    bca_ci, z0, acc = bca(b10, loo, s["point"])
    rows.append(dict(statistic=s["name"], universe=s["universe"], N_pairs=s["N"],
                     point=s["point"], paper_ci=s["paper_ci"], paper_lo=s["plo"], paper_hi=s["phi"],
                     b1000_lo=float(pct1k[0]), b1000_hi=float(pct1k[1]),
                     b10000_lo=float(pct10k[0]), b10000_hi=float(pct10k[1]),
                     bca_lo=bca_ci[0], bca_hi=bca_ci[1], z0=z0, accel=acc))

# HHI rows
hhi_sorted = np.sort(hhi_multi)
n = n_hhi
loo_mean = (hhi_multi.sum() - hhi_multi) / (n - 1)
m1 = n // 2  # n even? 11694 even -> median of n-1=11693 (odd) = element (n-1)//2 = 5846
k = (n - 1) // 2
loo_med = np.where(np.arange(n) <= k, hhi_sorted[np.minimum(k + 1, n - 1)], hhi_sorted[k]) if n % 2 == 0 else None
# careful: n=11694 even, n-1=11693 odd -> LOO median = sorted-without-i element at index (n-2)/2=5846
k = (n - 2) // 2
loo_med = np.where(np.arange(n) <= k, hhi_sorted[k + 1], hhi_sorted[k])
for name, point, boots, loo, pc, plo, phi in [
        ("HHI_mean_multi", hhi_mean_point, boot_hhi_mean, loo_mean, "[0.602,0.608]", 0.602, 0.608),
        ("HHI_median_multi", hhi_med_point, boot_hhi_med, loo_med, "[0.556,0.577]", 0.556, 0.577)]:
    pct1k = np.percentile(boots[:1000], [2.5, 97.5])
    pct10k = np.percentile(boots, [2.5, 97.5])
    bca_ci, z0, acc = bca(boots, loo, point)
    rows.append(dict(statistic=name, universe=f"iid over {n:,} multi-agent consumers", N_pairs=n,
                     point=point, paper_ci=pc, paper_lo=plo, paper_hi=phi,
                     b1000_lo=float(pct1k[0]), b1000_hi=float(pct1k[1]),
                     b10000_lo=float(pct10k[0]), b10000_hi=float(pct10k[1]),
                     bca_lo=bca_ci[0], bca_hi=bca_ci[1], z0=z0, accel=acc))

res = pd.DataFrame(rows)
res["paper_within_b10000_pct"] = (res["b10000_lo"] <= res["paper_lo"] + 1e-12) & (res["b10000_hi"] >= res["paper_hi"] - 1e-12) | res["paper_lo"].isna()
res["paper_within_bca"] = (res["bca_lo"] <= res["paper_lo"] + 1e-12) & (res["bca_hi"] >= res["paper_hi"] - 1e-12) | res["paper_lo"].isna()
res.to_csv(BASE + "vr28b_bootstrap_ci_table_20260906.csv", index=False)

elapsed = time.time() - t0
R = dict(script=SCRIPT, seed=SEED, B=B, elapsed_sec=elapsed,
         rows=rows)
with open(BASE + "vr28b_bootstrap_20260906.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = res[["statistic", "point", "paper_ci", "b1000_lo", "b1000_hi", "b10000_lo", "b10000_hi", "bca_lo", "bca_hi"]]
print(show.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
ru = res[res["statistic"] == "R_u_MoR"].iloc[0]
print(f"\n[专项核验] R_u 点估计 0.9807 -> B=1000 CI=[{ru['b1000_lo']:.4f},{ru['b1000_hi']:.4f}] vs R3信锚 [0.9792,0.9822]"
      f" | B=10000 CI=[{ru['b10000_lo']:.4f},{ru['b10000_hi']:.4f}] | BCa=[{ru['bca_lo']:.4f},{ru['bca_hi']:.4f}]")
print(f"[{SCRIPT}] DONE elapsed={elapsed:.0f}s", flush=True)
