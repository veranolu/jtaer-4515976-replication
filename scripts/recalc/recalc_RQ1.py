# -*- coding: utf-8 -*-
"""RQ1 锁定指标独立复算：75,411/86.6%/0.947/0.6050/76,829/1.17/0.49/13/50.8/471"""
import pandas as pd, numpy as np
df = pd.read_csv("/Coze/Drive/扣子/所有对话/主对话/论文5B.2_JTAER/full_686k_triplets.csv")
df.columns = ["user_id","agent_id","item_id"]

apc = df.groupby("user_id")["agent_id"].nunique()          # 每消费者 agent 数
n_single = int((apc==1).sum()); n_multi = int((apc>=2).sum())
print(f"单agent消费者={n_single} (主文写75,412→应为{n_single}) | 占比={n_single/len(apc)*100:.2f}% (锚86.6%)")
print(f"多agent消费者={n_multi} (锚11,694) | agents/consumer mean={apc.mean():.2f} SD={apc.std():.2f} max={apc.max()} (锚1.17/0.49/13)")

cnt = df.groupby(["user_id","agent_id"]).size()            # cross-agent HHI
sh = cnt / cnt.groupby("user_id").transform("sum")
hhi = (sh**2).groupby("user_id").sum()
print(f"HHI 全样本 mean={hhi.mean():.4f} (锚0.947) | multi-agent mean={hhi[apc>=2].mean():.4f} (锚0.6050) median={hhi[apc>=2].median():.4f} (锚0.5634)")

print(f"HHI>0.80 消费者={int((hhi>0.80).sum())} (锚76,829) | 占比={(hhi>0.80).mean()*100:.2f}% (锚88.2%)")

ac = df.groupby("agent_id").size()
print(f"transactions/agent mean={len(df)/df.agent_id.nunique():.1f} (锚50.8) | top-agent={int(ac.max())} (锚471) 占比={ac.max()/len(df)*100:.2f}% (锚0.07%)")
