"""R3 extension (declared after the primary run): soft-tissue denominator for all primary species.

Soft tissue = body - cuticle. Source per specimen: owners' S1 (as in the preregistered R3) where available, otherwise
our run of the owners' own segmentation program (Matte et al. 2025, vendored unchanged; stage cuticle_matte v2/v2b).
Threshold rule, fixed before looking at any R3 estimate: of the two thresholds in the authors' scripts (110, 145), use
the one with the smaller median |log10(ours / S1)| for cuticle volume on the S1 validation specimens; 130 is diagnostic.
Specimen QC rule (ours only; S1 untouched), fixed from segmentation diagnostics before the final fits: (a) body volume
stable across the three thresholds, max/min <= 1.15 with all > 0 (fails where the medium is as bright as cuticle and
the vial fills the mask, mostly 6.11 um scans); (b) cuticle/body at the chosen threshold within [0.03, 0.40].
REVISION (16 Nov, before the final fits and without looking at any R3 estimate under it): rule (a) removed half of the
2.44 um targets because t110 leaks into a bright medium while t130 and t145 agree (the authors' scripts carry both 110
and 145, i.e. they switched by scan). Main rule is now ADAPTIVE: per specimen use t110 if body(t110)/body(t130) is
within 1/1.10..1.10, else t145 if body(t145)/body(t130) is within 1/1.10..1.10, else exclude; then cuticle/body at
the used threshold within [0.03, 0.40]. The original strict rule is kept as a sensitivity analysis.
Disclosure: an interim R3x estimate on 28 added species had been seen before this rule was written (soft beta2 ~ +0.007).
Usage: r3x_run.py <cuticle jsonl (all v2 + v2b records)>
"""
import json, sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, pandas as pd
from common import *
import stages.brain_analysis as B

UM3 = 1e9
rec = pd.DataFrame([json.loads(l) for l in open(sys.argv[1], encoding="utf-8")])
rec = rec[rec.status == "ok"].drop_duplicates("key", keep="last").set_index("key")
B3 = rec[["t110_body_um3", "t130_body_um3", "t145_body_um3"]]
rec["stable"] = (B3.min(axis=1) > 0) & (B3.max(axis=1) / B3.min(axis=1).where(B3.min(axis=1) > 0) <= 1.15)
spec = pd.read_csv(RUN_OUT / "specimens_table.csv", dtype={"vox_class": str})
df, tree = primary_data(), mcc()
t0, _ = B.fit_primary(df, tree, SESOI); assert abs(t0["estimate"] - 0.0351) < 5e-4 and t0["n"] == 270, t0
out = {"n_records_ok": int(len(rec))}

# 1. validation on S1 specimens measured by us
v_all = spec[spec.key.isin(rec.index) & spec.s1_cuticle_um3.notna()].copy()
v = v_all[v_all.key.map(rec.stable)].copy()
out["validation_n_all"] = int(len(v_all)); out["validation_unstable"] = sorted(set(v_all.key) - set(v.key))
val = {}
for t in ("t110", "t130", "t145"):
    b, c = v.key.map(rec[f"{t}_body_um3"]), v.key.map(rec[f"{t}_cuticle_um3"])
    s1b, s1c = v.s1_body_um3, v.s1_cuticle_um3
    so, s1s = b - c, s1b - s1c
    lr = lambda x, y: np.log10(x / y)
    val[t] = {"n": int(len(v)),
              "body_median_ratio": float(np.median(b / s1b)), "cuticle_median_ratio": float(np.median(c / s1c)),
              "soft_median_ratio": float(np.median(so / s1s)),
              "cuticle_median_abs_log": float(np.median(abs(lr(c, s1c)))), "soft_median_abs_log": float(np.median(abs(lr(so, s1s)))),
              "r_log_body": float(np.corrcoef(np.log10(b), np.log10(s1b))[0, 1]),
              "r_log_cuticle": float(np.corrcoef(np.log10(c), np.log10(s1c))[0, 1]),
              "r_log_soft": float(np.corrcoef(np.log10(so), np.log10(s1s))[0, 1]),
              "r_cuticle_fraction": float(np.corrcoef(c / b, s1c / s1b)[0, 1])}
out["validation"] = val
tag = min(("t110", "t145"), key=lambda t: val[t]["cuticle_median_abs_log"]); out["threshold_chosen"] = tag
frac = rec[f"{tag}_cuticle_um3"] / rec[f"{tag}_body_um3"]
rec["qc_pass_strict"] = rec.stable & frac.between(0.03, 0.40)
within = lambda a, b: (a > 0) & (b > 0) & ((a / b.where(b > 0)).between(1 / 1.10, 1.10))
use110 = within(rec.t110_body_um3, rec.t130_body_um3); use145 = ~use110 & within(rec.t145_body_um3, rec.t130_body_um3)
rec["thr_used"] = np.where(use110, "t110", np.where(use145, "t145", "none"))
for q in ("body", "cuticle"):
    rec[f"ad_{q}_um3"] = np.where(use110, rec[f"t110_{q}_um3"], np.where(use145, rec[f"t145_{q}_um3"], np.nan))
rec["qc_pass"] = (rec.thr_used != "none") & (rec.ad_cuticle_um3 / rec.ad_body_um3).between(0.03, 0.40)
va = v_all[v_all.key.map(rec.qc_pass)]
b_, c_ = va.key.map(rec.ad_body_um3), va.key.map(rec.ad_cuticle_um3)
out["validation_adaptive"] = {"n": int(len(va)), "n_t145": int((va.key.map(rec.thr_used) == "t145").sum()),
    "body_median_ratio": float(np.median(b_ / va.s1_body_um3)), "cuticle_median_ratio": float(np.median(c_ / va.s1_cuticle_um3)),
    "soft_median_ratio": float(np.median((b_ - c_) / (va.s1_body_um3 - va.s1_cuticle_um3))),
    "r_log_soft": float(np.corrcoef(np.log10(b_ - c_), np.log10(va.s1_body_um3 - va.s1_cuticle_um3))[0, 1]),
    "r_log_body": float(np.corrcoef(np.log10(b_), np.log10(va.s1_body_um3))[0, 1]),
    "excluded": sorted(set(v_all.key) - set(va.key))}
out["qc"] = {"n": int(len(rec)), "stable": int(rec.stable.sum()), "pass_strict": int(rec.qc_pass_strict.sum()), "pass": int(rec.qc_pass.sum()),
             "thr_used": rec[rec.qc_pass].thr_used.value_counts().to_dict(),
             "fail_by_voxel": rec[~rec.qc_pass].voxel_um.value_counts().to_dict(), "n_by_voxel": rec.voxel_um.value_counts().to_dict()}

# 2. specimen soft tissue: S1 > ours; same exclusions as the preregistered R3 (body damage -> missing)
def species_soft(spec, tag, ours_for_all=False):
    s = spec[spec.main_ok & spec.log10_body.notna()].copy()
    if tag == "adaptive":
        good = rec[rec.qc_pass]; ours = s.key.map(good.ad_body_um3) - s.key.map(good.ad_cuticle_um3)
    else:
        good = rec[rec.qc_pass_strict]; ours = s.key.map(good[f"{tag}_body_um3"]) - s.key.map(good[f"{tag}_cuticle_um3"])
    s1 = s.s1_body_um3 - s.s1_cuticle_um3
    soft = ours if ours_for_all else s1.where(s1.notna(), ours)
    soft = soft.where(~s.excl_body_damage.astype(bool))
    s["src"] = np.where(s1.notna() & (not ours_for_all), "S1", np.where(ours.notna(), "ours", "none"))
    s["log10_soft_x"] = np.log10(soft.where(soft > 0) / UM3)
    s = s[s.stained.values == s.species_key.map(df.set_index("species_key").stain).values]
    g = s.groupby("species_key").agg(log10_soft_x=("log10_soft_x", "mean"), n_soft_x=("log10_soft_x", "count"),
                                     src=("src", lambda x: ",".join(sorted(set(x) - {"none"}))))
    return g

def fit(d, size):
    t, _ = B.fit_primary(d.reset_index(drop=True), tree, SESOI, size=size); return B._brief(t)

res = {}
d_pre = df.dropna(subset=["log10_soft"]); res["R3_prereg_S1_only"] = fit(d_pre, "log10_soft")
for t in ("adaptive", "t110", "t130", "t145"):
    g = species_soft(spec, t); d = df.merge(g, on="species_key", how="left").dropna(subset=["log10_soft_x"])
    res[f"R3x_{t}"] = {**fit(d, "log10_soft_x"), "n_species_S1": int((g.loc[d.species_key].src == "S1").sum()),
                       "n_species_ours_only": int((g.loc[d.species_key].src == "ours").sum())}
    res[f"R3x_{t}_body_same_species"] = fit(d, "log10_body")
# main extended estimate (adaptive rule) also over posterior trees
g = species_soft(spec, "adaptive"); d = df.merge(g, on="species_key", how="left").dropna(subset=["log10_soft_x"]).reset_index(drop=True)
pt, pool = B.primary_over_trees(d.rename(columns={"log10_body": "log10_body_orig", "log10_soft_x": "log10_body"}), posterior(), SESOI)
res["R3x_chosen_posterior_trees"] = {k: pool[k] for k in ("estimate_median", "pooled_ci_low", "pooled_ci_high", "share_p_lt_0.05", "outcome") if k in pool}
d["log10_cuticle_frac"] = np.log10(1 - 10 ** (d.log10_soft_x - d.log10_body))
res["corr_colony_softfrac"] = float(np.corrcoef(d.log10_colony_any, d.log10_soft_x - d.log10_body)[0, 1])
out["fits"] = res
(WORK / "r3x.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps(out, indent=1, default=float))
