# -*- coding: utf-8 -*-
"""overlap 复核：购买层 2.925% / 货架层 50.5% / 冗余率 0.0075 / 97.1%"""
import numpy as np, pandas as pd, time
from collections import Counter
t0 = time.time()
df = pd.read_csv("/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv")
df.columns = ["user_id","agent_id","item_id"]

user_agents = df.groupby("user_id")["agent_id"].apply(lambda s: np.array(sorted(set(s))))
multi_idx = user_agents[user_agents.apply(len) >= 2].index
n_multi = len(multi_idx)
print(f"[overlap] 多导购消费者={n_multi} (锚11,694) | 单agent={87105-n_multi} (主文写75,412)")

ui = df.groupby(["user_id","agent_id"])["item_id"].apply(set)
# 货架层: agent 的全部交易 item 集
shelf = df.groupby("agent_id")["item_id"].apply(set)

nz_purchase = 0; nz_shelf = 0; red_num = 0; red_den = 0
for uid in multi_idx:
    agents = user_agents.loc[uid]
    psets = [ui.loc[(uid, g)] for g in agents]
    # 购买层: 跨 agent 购买 item 重复
    cnt = Counter()
    for s in psets:
        for it in s: cnt[it] += 1
    dup_items = {it for it, v in cnt.items() if v >= 2}
    if dup_items: nz_purchase += 1
    # 冗余率口径: 重复购买的三元组数 / 总三元组数
    tot = sum(len(s) for s in psets)
    red_num += sum(v-1 for v in cnt.values() if v >= 2); red_den += tot
    # 货架层: 该消费者各 agent 的货架交集
    scnt = Counter()
    for g in agents:
        for it in shelf.loc[g]: scnt[it] += 1
    if any(v >= 2 for v in scnt.values()): nz_shelf += 1

print(f"[overlap] 购买层非零重叠={nz_purchase} ({nz_purchase/n_multi*100:.4f}%) (锚342=2.925%) | 零重叠={1-nz_purchase/n_multi:.4f} (锚97.1%)")
print(f"[overlap] 货架层非零重叠={nz_shelf} ({nz_shelf/n_multi*100:.4f}%) (锚5,911=50.5%)")
print(f"[overlap] 冗余率(粗口径, 重复unique项/总unique)={red_num/red_den:.6f} (锚0.0075)  ({time.time()-t0:.0f}s)")
