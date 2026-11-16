"""Exploratory companion to R2 (not preregistered): the same one-species-per-genus draws (same seed, same genus groups)
but each drawn species keeps its own tip in the Economo 2018 MCC tree. Separates the effect of the tree from the
effect of reducing the data to ~115 genera."""
import json, sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *
import numpy as np
import stages.brain_analysis as B
from analysis import robustness_extra as RX

bt = RX.read_nexus_tree(ROOT / "work/borowiec2025/10-MCMCTree/root-prior-129Ma-to-158Ma/independent/FigTree-root-prior-129to158Ma-independent.tre")
tree_e = mcc(); prim = primary_data().assign(tip_economo=lambda x: x.tip)
fit = lambda d, V: B.fit_primary(d, tree_e, SESOI, V=tree_e.vcv(list(d.tip_economo)))[0]
rep, cover = RX.borowiec_resample(prim, bt, fit, n_rep=100, seed=2025, transfers=pgls.GENUS_TRANSFERS)
s = RX.summarize_resamples(rep, SESOI)
rep.to_csv(WORK / "r2_economo_genus_reps.csv", index=False)
(WORK / "r2_economo_genus.json").write_text(json.dumps({"coverage": cover, "summary": s}, indent=2, default=float), encoding="utf-8")
print({x: s[x] for x in ("estimate_median", "estimate_q025", "estimate_q975", "pooled_ci_low", "pooled_ci_high", "share_p_lt_0.05", "outcome", "outcome_counts", "lambda_median", "n_species_min")})
