# -*- coding: utf-8 -*-
"""独立重算路径A：pair结构 + H_obs + E_u + E_g + coverage + S9 + overlap
对照 summary_T2.json / coverage_T25.csv / overlap_null_T24.csv"""
import numpy as np, pandas as pd, json, time
from scipy.special import gammaln
from collections import Counter

t0 = time.time()
TRIP = "/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv"
df = pd.read_csv(TRIP)
df.columns = ["user_id","agent_id","item_id"]
a = (len(df), df.user_id.nunique(), df.agent_id.nunique(), df.item_id.nunique())
assert a == (686246, 87105, 13508, 60562), a
print(f"[六锚] {a} ✅  ({time.time()-t0:.0f}s)")

pair_items = df.groupby(["user_id","agent_id"])["item_id"].apply(list)
ug = list(pair_items.index)
assert len(ug) == 101577
pair_user = np.array([k[0] for k in ug])
pair_agent = np.array([k[1] for k in ug])
pair_g = pair_items.apply(len).values.astype(np.int64)
guide_items = df.groupby("agent_id")["item_id"].apply(lambda s: np.array(sorted(set(s))))
pair_ns = np.array([len(guide_items[g]) for g in pair_agent], dtype=np.int64)
print(f"[pair] g mean={pair_g.mean():.4f} | g=1: {(pair_g==1).sum()} | g>=2: {(pair_g>=2).sum()}")

# H_obs
H_obs = np.zeros(len(ug))
for i, its in enumerate(pair_items):
    c = np.array(list(Counter(its).values()), dtype=np.float64)
    p = c / c.sum()
    H_obs[i] = -np.sum(p * np.log(p))
print(f"[H_obs] mean全={H_obs.mean():.6f} (锚1.4315) | g>=2={H_obs[pair_g>=2].mean():.6f} (锚1.6769)")

def e_uniform_exact(g, n_s):
    if g <= 1 or n_s <= 1: return 0.0
    k = np.arange(1, g+1, dtype=np.float64)
    p = 1.0/n_s
    logpmf = (gammaln(g+1)-gammaln(k+1)-gammaln(g-k+1) + k*np.log(p) + (g-k)*np.log1p(-p))
    t = (k/g)*np.log(k/g)
    return float(-n_s*np.sum(t*np.exp(logpmf)))

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

# S9 worked example
s9 = e_uniform_exact(5, 10)
print(f"[S9] E[H|g=5,n_shelf=10] = {s9:.6f} (锚1.349) | ln5={np.log(5):.4f} ln10={np.log(10):.4f} 占比={np.log(5)/np.log(10)*100:.2f}% (锚69.9%)")

# E_u（缓存）
cache = {}
E_u = np.empty(len(ug))
for i in range(len(ug)):
    key = (int(pair_g[i]), int(pair_ns[i]))
    v = cache.get(key)
    if v is None:
        v = e_uniform_exact(*key); cache[key] = v
    E_u[i] = v
print(f"[E_u] cells={len(cache)} | mean全={E_u.mean():.6f} (锚1.4400) | g>=2={E_u[pair_g>=2].mean():.6f} (锚1.6869)  ({time.time()-t0:.0f}s)")

# E_g：平台热度货架内归一化，(agent,g) cell 缓存
global_pop = df["item_id"].value_counts()
agent_ids = guide_items.index.values
a2i = {a_: i for i, a_ in enumerate(agent_ids)}
pa_pos = np.array([a2i[x] for x in pair_agent])
E_g = np.empty(len(ug))
cell_cache = {}
for i in range(len(ug)):
    key = (pa_pos[i], int(pair_g[i]))
    v = cell_cache.get(key)
    if v is None:
        items = guide_items[agent_ids[pa_pos[i]]]
        fq = global_pop.reindex(items).values.astype(np.float64)
        v = e_exact(fq/fq.sum(), int(pair_g[i]))
        cell_cache[key] = v
    E_g[i] = v
print(f"[E_g] cells={len(cell_cache)} | mean全={E_g.mean():.6f} (锚0.7842) | g>=2={E_g[pair_g>=2].mean():.6f} (锚0.9186)  ({time.time()-t0:.0f}s)")

np.savez("recalc_stage1.npz", pair_user=pair_user, pair_agent=pair_agent, pair_g=pair_g,
         pair_ns=pair_ns, H_obs=H_obs, E_u=E_u, E_g=E_g,
         ug=np.array([(u,g) for u,g in ug]))
print(f"[落盘] recalc_stage1.npz  ({time.time()-t0:.0f}s)")

# ============ coverage（缺口1 + 对照）============
n_shelf = pair_ns.astype(np.float64)
E_uniq = np.where(n_shelf > 0, n_shelf*(1.0-(1.0-1.0/n_shelf)**pair_g), np.nan)
pair_nd = np.array([len(set(its)) for its in pair_items], dtype=np.float64)
rom  = pair_nd.mean()/E_uniq.mean()
mor  = np.mean(pair_nd/E_uniq)
g2 = pair_g >= 2
print(f"[coverage] mean_obs_unique={pair_nd.mean():.6f} | mean_exp_unique={E_uniq.mean():.6f} (N=101,577 全样本)")
print(f"[coverage] g>=2: mean_obs={pair_nd[g2].mean():.6f} | mean_exp={E_uniq[g2].mean():.6f}")
print(f"[coverage] ratio_of_means={rom:.6f} (锚1.0289) | mean_of_pair_ratios={mor:.6f} (锚1.0040)")
print(f"[coverage] g>=2: RoM={pair_nd[g2].mean()/E_uniq[g2].mean():.6f} | MoR={np.mean(pair_nd[g2]/E_uniq[g2]):.6f}")
np.savez("recalc_coverage.npz", pair_nd=pair_nd, E_uniq=E_uniq)

# ============ overlap 复核（2.925% / 50.5% / 0.0075）============
# 多导购消费者
user_agents = df.groupby("user_id")["agent_id"].agg(lambda s: np.array(sorted(set(s))))
multi = user_agents[user_agents.apply(len) >= 2]
n_multi = len(multi)
print(f"[overlap] 多导购消费者={n_multi} (锚11,694) | 单agent={87105-n_multi} (锚75,412?)")

# 购买层：每消费者在各 agent 处购买的 item 集合的跨 agent 交集
ui = df.groupby(["user_id","agent_id"])["item_id"].agg(set)
nz_purchase = 0
nz_shelf = 0
redundant_rates = []  # 冗余率: 重复购买次数/总购买次数?
for uid in multi.index:
    sets = [ui.loc[(uid, g)] for g in multi.loc[uid]]
    # 两两并集交集: 任意两 agent item 集有交
    inter_all = set.intersection(*sets) if len(sets) > 1 else set()
    # 观测口径: 存在任意一对 agent 购买同一 item
    union_cnt = Counter()
    for s in sets:
        for it in s: union_cnt[it] += 1
    has_overlap = any(v >= 2 for v in union_cnt.values())
    if has_overlap: nz_purchase += 1
print(f"[overlap] 购买层非零重叠消费者={nz_purchase} | 比例={nz_purchase/n_multi*100:.4f}% (锚2.925%, 342)")
print(f"  342/11694={342/11694*100:.4f}% | 97.1%零重叠对照: {1-nz_purchase/n_multi:.4f}")
