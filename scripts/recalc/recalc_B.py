# -*- coding: utf-8 -*-
"""路径B：E_s LOO 独立重算 + ratio统计对照 + bootstrap CI(缺口2/3)"""
import numpy as np, pandas as pd, time, json
from scipy.special import gammaln
from collections import Counter
t0 = time.time()

df = pd.read_csv("/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv")
df.columns = ["user_id","agent_id","item_id"]
d = np.load("recalc_stage1.npz")
pair_user, pair_agent, pair_g, pair_ns = d["pair_user"], d["pair_agent"], d["pair_g"], d["pair_ns"]
H_obs, E_u, E_g = d["H_obs"], d["E_u"], d["E_g"]
n_pairs = len(pair_g)

def e_exact(p, g):
    p = np.asarray(p, dtype=np.float64)
    if g <= 1 or len(p) <= 1: return 0.0
    if (p >= 1.0-1e-15).any(): return 0.0
    p = p[p > 0]
    if len(p) <= 1: return 0.0
    k = np.arange(1, g+1, dtype=np.float64)
    logbin = gammaln(g+1)-gammaln(k+1)-gammaln(g-k+1)
    lp = np.log(p); lq = np.log1p(-p)
    M = np.exp(np.outer(k, lp) + np.outer(g-k, lq))
    S = M.sum(axis=1)
    t = (k/g)*np.log(k/g)
    return float(-np.sum(t*np.exp(logbin)*S))

pair_items = df.groupby(["user_id","agent_id"])["item_id"].apply(list)
ug = list(pair_items.index)
guide_items = df.groupby("agent_id")["item_id"].apply(lambda s: np.array(sorted(set(s))))
ai_cnt = df.groupby(["agent_id","item_id"]).size()
agent_cnt = {gid: ai_cnt.loc[gid].to_dict() for gid in guide_items.index}

# E_u 缓存复用（fallback 用）
cache_u = {}
def eu(g, n):
    key = (int(g), int(n))
    v = cache_u.get(key)
    if v is None:
        if g <= 1 or n <= 1: v = 0.0
        else:
            k = np.arange(1, g+1, dtype=np.float64)
            p = 1.0/n
            logpmf = (gammaln(g+1)-gammaln(k+1)-gammaln(g-k+1)+k*np.log(p)+(g-k)*np.log1p(-p))
            t = (k/g)*np.log(k/g)
            v = float(-n*np.sum(t*np.exp(logpmf)))
        cache_u[key] = v
    return v

E_s = np.empty(n_pairs)
loo_fb = np.zeros(n_pairs, dtype=bool)
for i in range(n_pairs):
    gid = pair_agent[i]
    shelf = guide_items[gid]
    pc = Counter(pair_items.iloc[i])
    ac = agent_cnt[gid]
    loo = np.fromiter((ac.get(it,0)-pc.get(it,0) for it in shelf), dtype=np.float64, count=len(shelf))
    s = loo.sum()
    if s <= 0:
        loo_fb[i] = True
        E_s[i] = eu(pair_g[i], pair_ns[i])
    else:
        E_s[i] = e_exact(loo/s, int(pair_g[i]))
    if (i+1) % 20000 == 0:
        print(f"  E_s {i+1:,}/{n_pairs:,} ({time.time()-t0:.0f}s)", flush=True)
print(f"[E_s] mean全={E_s.mean():.6f} (锚1.3907) | g>=2={E_s[pair_g>=2].mean():.6f} (锚1.6291)")
print(f"[LOO fallback] {loo_fb.sum()} (锚0) | g>=2内: {(loo_fb&(pair_g>=2)).sum()} (锚0)")
np.savez("recalc_stage2.npz", E_s=E_s, loo_fb=loo_fb)

# ============ ratio 统计对照（g>=2, E>0 掩膜）============
g2 = pair_g >= 2
masks = {
    "R_u": g2 & (E_u > 0),
    "R_s": g2 & (E_s > 0),
    "R_g": g2 & (E_g > 0),
}
R = {"R_u": H_obs/E_u, "R_s": H_obs/E_s, "R_g": H_obs/E_g}
expect = {
    "R_u": dict(mean=0.9807, ci=(0.9792,0.9822), med=1.0361, mc=(1.0359,1.0365), n=86533),
    "R_s": dict(mean=1.0169, ci=(1.0148,1.0189), med=1.0473, mc=(1.0470,1.0476), n=86495),
    "R_g": dict(mean=2.4337, ci=(2.4160,2.4539), med=1.8972, mc=(1.8879,1.9069), n=86533),
}
print("\n===== pair-level ratio 对照 =====")
for k in ["R_u","R_s","R_g"]:
    m = masks[k]
    v = R[k][m]
    e = expect[k]
    print(f"{k}: N={m.sum()} (锚{e['n']}) | mean={v.mean():.4f} (锚{e['mean']}) | median={np.median(v):.4f} (锚{e['med']})")

# ratio of means
print("\n===== ratio of means 对照 =====")
rom_u = H_obs[g2].mean()/E_u[g2].mean()
rom_s = H_obs[g2].mean()/E_s[g2].mean()
rom_g = H_obs[g2].mean()/E_g[g2].mean()
print(f"uniform RoM={rom_u:.6f} (锚0.9941) | popshelf RoM={rom_s:.6f} (锚1.0294) | popglobal RoM={rom_g:.6f} (锚1.8255)")

# J_g
J = np.where(g2, H_obs/np.log(np.maximum(pair_g,2)), np.nan)
print(f"J_g mean={np.nanmean(J):.4f} (锚0.9269) | median={np.nanmedian(J):.4f} (锚1.0)")

# ============ bootstrap CI（consumer-level, 1000, seed 42）============
print("\n===== bootstrap CI 导出 =====")
users_unique = np.unique(pair_user)
user_to_idx = {}
for i, u in enumerate(pair_user):
    user_to_idx.setdefault(u, []).append(i)
maxuid = users_unique.max()
uid_map = np.zeros(maxuid+1, dtype=np.int64) - 1
uid_map[users_unique] = np.arange(len(users_unique))
pair_uidx = uid_map[pair_user]

rng = np.random.RandomState(42)
B = 1000
dc = np.load("recalc_coverage.npz")
pair_nd, E_uniq = dc["pair_nd"], dc["E_uniq"]

def boot_rom(numer, denom, mask=None):
    """ratio of means 的 consumer-level bootstrap CI"""
    if mask is None: mask = np.ones(n_pairs, dtype=bool)
    idx_all = np.where(mask)[0]
    puidx = pair_uidx[idx_all]
    out = np.empty(B)
    n_users = len(users_unique)
    for b in range(B):
        su = rng.randint(0, n_users, size=n_users)
        w = np.bincount(su, minlength=n_users)[puidx]
        nw = w.sum()
        out[b] = (numer[idx_all]*w).sum()/nw / ((denom[idx_all]*w).sum()/nw)
    return np.percentile(out, [2.5, 97.5]), out.mean()

# 缺口2: coverage RoM CI（全样本 N=101,577 + g>=2）
ci_cov_all, m_cov_all = boot_rom(pair_nd, E_uniq)
ci_cov_g2, m_cov_g2 = boot_rom(pair_nd, E_uniq, g2)
print(f"[缺口2] coverage RoM CI 全样本: [{ci_cov_all[0]:.4f}, {ci_cov_all[1]:.4f}] (点估计 1.0289)")
print(f"[缺口2] coverage RoM CI g>=2:   [{ci_cov_g2[0]:.4f}, {ci_cov_g2[1]:.4f}] (点估计 1.0297)")

# 缺口3: pop_global RoM CI (g>=2)
ci_pg, m_pg = boot_rom(H_obs, E_g, g2)
print(f"[缺口3] pop_global RoM CI g>=2: [{ci_pg[0]:.4f}, {ci_pg[1]:.4f}] (点估计 1.8255)")
# 附加: uniform / pop_shelf RoM CI（一致性参考）
ci_u, _ = boot_rom(H_obs, E_u, g2)
ci_s, _ = boot_rom(H_obs, E_s, g2)
print(f"[参考] uniform RoM CI: [{ci_u[0]:.4f}, {ci_u[1]:.4f}] | pop_shelf RoM CI: [{ci_s[0]:.4f}, {ci_s[1]:.4f}]")

# pair-level ratio mean CI 复核（对照 summary boot_ci_42）
def boot_ratio_mean(v, mask):
    idx_all = np.where(mask)[0]
    puidx = pair_uidx[idx_all]
    out = np.empty(B)
    n_users = len(users_unique)
    for b in range(B):
        su = rng.randint(0, n_users, size=n_users)
        w = np.bincount(su, minlength=n_users)[puidx]
        out[b] = (v[idx_all]*w).sum()/w.sum()
    return np.percentile(out, [2.5, 97.5])
rng2 = np.random.RandomState(42)
rng = rng2
ci_ru = boot_ratio_mean(R["R_u"], masks["R_u"])
print(f"[复核] R_u mean CI=[{ci_ru[0]:.4f},{ci_ru[1]:.4f}] (锚[0.9792,0.9822])")

print(f"\n全部完成 ({time.time()-t0:.0f}s)")
