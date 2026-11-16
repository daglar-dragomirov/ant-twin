"""R2 (prereg sec. 8): primary model on the Borowiec et al. 2025 dated UCE tree (Zenodo 14338504, 10-MCMCTree).
The tree samples ~one exemplar per genus, so (owners' rule) one species per genus is drawn at random, 100 resamples;
we report the median and spread of beta2, the pooled t-mixture 95% interval and the share of resamples per outcome.
VCV is taken from the MRCA of the sampled tips (outgroup stem removed, as ape::keep.tip would)."""
import json, sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *
import numpy as np
import stages.brain_analysis as B
from analysis import robustness_extra as RX

BT = ROOT / "work/borowiec2025/10-MCMCTree"
TREES = {"129-158Ma_independent (main)": "root-prior-129Ma-to-158Ma/independent/FigTree-root-prior-129to158Ma-independent.tre",
         "129-158Ma_correlated": "root-prior-129Ma-to-158Ma/correlated/FigTree-root-prior-129to158Ma-correlated.tre",
         "under224Ma_independent": "root-prior-under224Ma/independent/FigTree-root-prior-under224Ma-independent.tre",
         "under224Ma_correlated": "root-prior-under224Ma/correlated/FigTree-root-prior-under224Ma-correlated.tre"}

sp = pd.read_csv(RUN_OUT / "species_table.csv", dtype={"vox_class": str})
pool272 = sp.dropna(subset=["log10_brain", "log10_body", "log10_colony_any"]).reset_index(drop=True)
prim = primary_data()
chk = prim.merge(pool272, on="species_key", suffixes=("", "_s"))
assert len(chk) == 270 and np.allclose(chk.log10_brain, chk.log10_brain_s) and np.allclose(chk.log10_colony_any, chk.log10_colony_any_s)


def fit(d, V):
    off = V[~np.eye(len(V), dtype=bool)].min()  # depth of the MRCA of the sampled tips
    return B.fit_primary(d, None, SESOI, V=V - off)[0]


out = {}
for name, f in TREES.items():
    tree = RX.read_nexus_tree(BT / f)
    for setname, d in (("272 species (all with brain+body+colony)", pool272), ("270 primary species", prim)):
        if not name.endswith("(main)") and setname.startswith("270"):
            continue
        for nrep in ((100, 1000) if name.endswith("(main)") and setname.startswith("272") else (100,)):
            rep, cover = RX.borowiec_resample(d, tree, fit, n_rep=nrep, seed=2025, transfers=pgls.GENUS_TRANSFERS)
            s = RX.summarize_resamples(rep, SESOI)
            k = f"{name} | {setname} | {nrep} resamples"
            out[k] = {"coverage": cover, "summary": s}
            rep.to_csv(WORK / f"r2_reps_{len(out)}.csv", index=False)
            print(k, {x: s[x] for x in ("estimate_median", "estimate_q025", "estimate_q975", "pooled_ci_low", "pooled_ci_high",
                                        "share_p_lt_0.05", "outcome", "outcome_counts", "lambda_median", "n_species_min")},
                  cover, flush=True)
(WORK / "r2.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
