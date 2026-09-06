# Anonymous Replication Package — jtaer-4515976 (VR28 audit core)

Minimal anonymized package for reproducing the key derived numbers audited in the
VR28 verification round (2026-09-06). 

## Contents

```
data/
  user_level_6vars_anonymized_87105.csv   # 87,105 rows, one per consumer
scripts/
  recalc/   # independent recalculation paths (VR23 round)
    recalc_RQ1.py   recalc_A.py   recalc_A2.py   recalc_B.py   recalc_S7.py
  vr28/     # VR28 round-1 + round-2 audit scripts
    vr28_audit_script.py  vr28_pairs_script.py  vr28_consumers_gge2.py
    vr28_remaining_script.py  vr28_final_checks.py  vr28_build_audit_csv.py
    vr28b_pair_table_20260906.py  vr28b_static_item1_20260906.py
    vr28b_static_v2_20260906.py  vr28b_segments_20260906.py
    vr28b_seg_variants_20260906.py  vr28b_mc_lopo_20260906.py
    vr28b_poporig_support_20260906.py  vr28b_bootstrap_20260906.py
README.md
```

## De-identification

- `user_level_6vars_anonymized_87105.csv` is derived from the internal six-variable
  user-level table by **dropping the `user_id` column**. Remaining columns are
  behavioural aggregates only: `n_purchases, n_agents, n_items, HHI, Entropy, TopShare`.
- No raw transaction triplets, agent ids, item ids, timestamps, or free text are included.
- Row order is preserved from the source table (sorted by the dropped id); rows are not
  individually linkable to any external identifier.

## Input data required by the scripts (NOT included)

The scripts expect the full anonymized transaction triplets file
`full_686k_triplets.csv` (686,246 rows; columns `user_id, agent_id, item_id`,
integer pseudonymous ids) at the path constants defined at the top of each script.
Adjust the `TRIP` / `BASE` path constants to your local layout before running.
Several scripts also read intermediate npz artifacts produced by `recalc_A.py` /
`recalc_B.py` (`recalc_stage1.npz`, `recalc_stage2.npz`, `recalc_coverage.npz`);
run the recalc scripts first to regenerate them.

## Reproduction steps

1. `python3 scripts/recalc/recalc_RQ1.py` — HHI / entropy / concentration anchors.
2. `python3 scripts/recalc/recalc_A.py` — pair structure, H_obs, E_u, E_g, coverage,
   overlap observation anchors; writes `recalc_stage1.npz`, `recalc_coverage.npz`.
3. `python3 scripts/recalc/recalc_B.py` — E_s (leave-one-out) + ratio statistics +
   consumer-level bootstrap CIs (B=1,000); writes `recalc_stage2.npz`.
4. `python3 scripts/recalc/recalc_A2.py`, `recalc_S7.py` — overlap & segment-set checks.
5. `python3 scripts/vr28/vr28b_pair_table_20260906.py` — rebuilds the 101,577-row
   pair-level table and all §4.2–4.3 statistics (dual-path vs recalc npz).
6. `python3 scripts/vr28/vr28b_mc_lopo_20260906.py` and
   `python3 scripts/vr28/vr28b_poporig_support_20260906.py` — Monte Carlo null
   benchmarks for cross-agent overlap (uniform / popularity × original / LOPO) and
   focal support coverage.
7. `python3 scripts/vr28/vr28b_bootstrap_20260906.py` — B=10,000 consumer-cluster
   bootstrap CI audit table (13 statistics; percentile + BCa).
8. Remaining `vr28*` scripts reproduce specific anchors (HHI histogram bins,
   consumer counts, static ratios, redundancy/segment statistics).

## Random seeds & environment

- **All stochastic procedures use seed = 42** (bootstrap: `np.random.RandomState(42)`;
  Monte Carlo null models: `np.random.default_rng(42)`; sensitivity seeds 3, 7, 11 are
  reported where applicable). Note the bootstrap and the MC nulls use different RNG
  streams than the original pipeline; agreement is therefore expected within Monte
  Carlo error, not to machine precision (observed ≤ 0.006 pp on overlap shares).
- Python 3.13, numpy 2.x, pandas 2.x, scipy 1.x. No other dependencies.
- Reference runtimes (cloud sandbox, 2026-09-06): pair table ~2 min; MC nulls ~70 s;
  B=10,000 bootstrap (13 statistics) ~17 s.

## Key definitions (audit conventions)

- Pair = consumer–agent cell with ≥1 purchase; g = purchases in the pair;
  n_s = distinct items on the agent's shelf; H_obs = observed Shannon entropy (nats).
- E_u: expected entropy under uniform draws from the pair shelf (exact binomial
  marginal computation). E_s: same under shelf popularity weights with the focal
  pair's own purchases left out (LOPO). E_g: same under platform-wide popularity
  weights restricted to the shelf. E_uniq: expected distinct items under uniform
  draws (coverage benchmark).
- R_u / R_s / R_g = mean of H_obs/E_* over pairs with g≥2 and defined denominators
  (n_s≥2 for R_u, R_g; E_s>0 for R_s): N = 86,533 / 86,495 / 86,533.
- Cross-agent overlap: consumer-level indicator that ≥2 of the consumer's agents
  sold the same item to that consumer (purchase layer); redundancy = |∩|/|∪| of
  purchased item sets, averaged over consumers.
