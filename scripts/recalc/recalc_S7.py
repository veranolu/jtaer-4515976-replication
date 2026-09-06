# -*- coding: utf-8 -*-
"""SM S7 独立复算：段定义 cat=item_id//10000（8 块，与 export_sm_figure_data_20260826.py L5 注释一致）
指标：91,226,278 对 / mean Jaccard 0.8997 / 全8段 78.5% / identical 61.8% / Ruzicka 0.5471(31.7%≤0.5) / consumer 500k 0.3430/13.3%"""
import numpy as np, pandas as pd, time
from itertools import combinations
t0 = time.time()
df = pd.read_csv("/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv")
df.columns = ["user_id","agent_id","item_id"]
df["cat"] = df["item_id"] // 10000                     # 段定义：8 个 ID 块
assert df["cat"].nunique() == 8, df["cat"].unique()

popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
agent_cats = df.groupby("agent_id")["cat"].agg(lambda s: set(s))
user_cats  = df.groupby("user_id")["cat"].agg(lambda s: set(s))
assert len(agent_cats) == 13_508 and len(user_cats) == 87_105

# ① 全8段占比
M_full = np.array([len(s) for s in agent_cats.values])
print(f"全8段 agent 占比={ (M_full==8).mean()*100:.2f}% (锚78.5%)")

# ② 全量 agent-pair Jaccard（256 掩码分组加权，精确非抽样）
am = np.array([sum(1 << int(c) for c in s) for s in agent_cats.values], dtype=np.uint8)
groups = pd.Series(am).value_counts()
tot_j, wtot, ident_w = 0.0, 0, 0.0
for m1, m2 in combinations(groups.index, 2):
    a, b = int(popcnt[m1 & m2]), int(popcnt[m1 | m2])
    w = int(groups[m1]) * int(groups[m2])
    tot_j += (a / b if b > 0 else 0.0) * w; wtot += w
for mk in groups.index:
    g = int(groups[mk])
    if g > 1:
        w = g * (g - 1) // 2
        tot_j += 1.0 * w; ident_w += w; wtot += w
print(f"agent-pair exact Jaccard mean={tot_j/wtot:.4f} (锚0.8997) | 对数={wtot:,} (锚91,226,278)")
print(f"identical 段集占比={ident_w/wtot*100:.2f}% (锚61.8%)")

# ③ Ruzicka：段份额向量 Σmin/Σmax（分块矩阵，Σmax=2−Σmin 因行和=1）
S = np.zeros((len(agent_cats), 8))
for r, gid in enumerate(agent_cats.index):
    v = df.loc[df.agent_id == gid, "cat"].value_counts()
    S[r, v.index.astype(int)] = v.values / v.values.sum()
ru_sum, cnt_le = 0.0, 0
n = len(S); B = 256
for i in range(0, n, B):
    D = S[i:i+B]
    m = np.minimum(D[:, None, :], S[None, :, :]).sum(axis=2)   # Σmin；Σmax=2−Σmin（行和=1）
    R = m / (2.0 - m)
    ru_sum += R.sum()                  # 全矩阵含对角 R_ii=1
    cnt_le += int((R <= 0.5).sum())    # 对角=1 不影响 ≤0.5 计数
ru_mean = (ru_sum - n) / (n * (n - 1)) # 扣 n 个对角项转无对均值
print(f"Ruzicka mean={ru_mean:.4f} (锚0.5471) | ≤0.5 占比={cnt_le/(n*(n-1))*100:.2f}% (锚31.7%)")

# ④ consumer 500k seed=42（先 agent 后 consumer 双段抽样，复现原顺序）
rng = np.random.default_rng(42)
NS = 500_000
_i1 = rng.integers(0, len(am), NS); _i2 = rng.integers(0, len(am), NS)
um = np.array([sum(1 << int(c) for c in s) for s in user_cats.values], dtype=np.uint8)
i1 = rng.integers(0, len(um), NS); i2 = rng.integers(0, len(um), NS)
inter = popcnt[um[i1] & um[i2]].astype(np.int16); union = popcnt[um[i1] | um[i2]].astype(np.int16)
jac = np.where(union > 0, inter / union, 0.0)
print(f"consumer 500k Jaccard mean={jac.mean():.6f} (锚0.3430) | 零共享={(inter==0).mean()*100:.2f}% (锚13.3%)")
print(f"全部完成 ({time.time()-t0:.0f}s)")
