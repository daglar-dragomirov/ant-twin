"""Supplementary computations for the preprint draft (exploratory; NOT part of the frozen results files).

Uses the frozen analysis code read-only (ant-twin `src/stages/brain_analysis.py`, `src/analysis/pgls.py`, branch
orx/brain-analysis-v1-primary-unblinded-preregistere, local worktree C:/projects/ant-twin-wt/analysis) and the local
copies of the primary-run outputs (work/robustness/run_outputs/) and of the R3-extension input (cuticle_final.jsonl).

Computes:
  1. R1 on all 200 Economo 2018 FBD posterior trees (crown 100 + stem 100). The primary run used 6 trees
     (config robustness.max_trees = 6); the preregistration names 200.
  2. R12 "owners' method" (mean VCV of the FBD trees / 200 as a fixed correlation) on all 200 trees.
  3. Body denominator on the 154 S1 species of the preregistered R3 (reproduces primary_v1.md, +0.020).
  4. Lambda-matched body vs soft-tissue comparison on the R3-extension species sets (adaptive 235, strict 197):
     both fits with the same fixed Pagel lambda (0, the body-fit ML value, 1).
  5. Species-level cuticle fraction table for Fig. 4b (same species-soft rule as scripts/robustness_v1/r3x_run.py).
  6. Body-volume quality sensitivity of the primary model (species means rebuilt with B.species_means after a specimen
     filter): drop specimens with brain/body > 10% (the preregistered plausibility flag range 0.05-10%, used here as an
     exclusion), > 5%, all autoflagged specimens; owners' S1 bodies only; own v8 bodies only.
  7. Like-for-like decomposition: body AND cuticle from the same instrument on the same specimens (S1 where measured,
     otherwise the owners' program under the adaptive QC rule) -> body vs soft-tissue denominators on the same species.
Output: supp_checks.json and species_soft_adaptive.csv next to this script.
Run:  uv run --no-project --python 3.13 --with pandas,numpy,scipy,boto3,openpyxl python supp_checks.py
"""
import sys, json, time, pathlib
sys.path.insert(0, "C:/projects/ant-twin-wt/analysis/src")
import numpy as np, pandas as pd
from analysis import pgls
import stages.brain_analysis as B

ROOT = pathlib.Path("C:/projects/dsc-biology/research/ant-twin")
RO = ROOT / "work/robustness/run_outputs"
TD = ROOT / "work/economo2018/inner/Dryad_archive/15k_all_ant_trees"
OUT = pathlib.Path(__file__).resolve().parent
SESOI = 0.02
KEEP = ("estimate", "se", "ci_low", "ci_high", "p", "n", "lambda", "outcome", "ci90")

df = pd.read_csv(RO / "primary_data.csv", dtype={"vox_class": str})
spec = pd.read_csv(RO / "specimens_table.csv", dtype={"vox_class": str})
mcc = pgls.read_trees(TD / "15k_NCuniform_stem_mcc.tre")[0]
brief = lambda t: {k: t[k] for k in KEEP if k in t}
res = {}

t0, _ = B.fit_primary(df, mcc, SESOI)
assert abs(t0["estimate"] - 0.0350593) < 1e-6 and t0["n"] == 270, t0          # reproduces primary.json
res["primary_reproduced"] = brief(t0)

# 1-2. all 200 FBD trees
t = time.time()
fbd = [tr for f in ("15K_FBD_crown_posterior.tre", "15K_FBD_stem_posterior.tre") for tr in pgls.read_trees(TD / f)]
pt, pool = B.primary_over_trees(df, fbd, SESOI)
res["R1_fbd_200"] = {k: pool[k] for k in pool if k not in ("ci95",)}
res["R1_fbd_200"]["n_trees_crown"], res["R1_fbd_200"]["n_trees_stem"] = 100, 100
ok = [tr for tr in fbd if df.tip.isin(set(tr.tip_index)).all()]
C = sum(tr.vcv(list(df.tip)) for tr in ok) / len(ok) / 200.0; np.fill_diagonal(C, 1.0)
X, names, _ = B.model_design(df, colony="log10_colony_any")
r12 = pgls.term(pgls.gls(df.log10_brain.to_numpy(float), X, C, names), "log10_colony_any")
r12.update(B.classify(r12["estimate"], r12["se"], r12["df"], SESOI)); r12["n_trees"] = len(ok)
res["R12_owners_meanVCV_200"] = brief(r12) | {"n_trees": len(ok)}
res["seconds_fbd"] = round(time.time() - t, 1)

# 3. body denominator on the 154 S1 species
d154 = df.dropna(subset=["log10_soft"]).reset_index(drop=True)
res["S1_154_body"] = brief(B.fit_primary(d154, mcc, SESOI)[0])
res["S1_154_soft"] = brief(B.fit_primary(d154, mcc, SESOI, size="log10_soft")[0])
res["S1_154_corr_colony_softfrac"] = float(np.corrcoef(d154.log10_colony_any, d154.log10_soft - d154.log10_body)[0, 1])

# 4-5. R3 extension species sets (rule copied from scripts/robustness_v1/r3x_run.py)
rec = pd.DataFrame([json.loads(l) for l in open(ROOT / "work/robustness/cuticle_final.jsonl", encoding="utf-8")])
rec = rec[rec.status == "ok"].drop_duplicates("key", keep="last").set_index("key")
B3 = rec[["t110_body_um3", "t130_body_um3", "t145_body_um3"]]
rec["stable"] = (B3.min(axis=1) > 0) & (B3.max(axis=1) / B3.min(axis=1).where(B3.min(axis=1) > 0) <= 1.15)
rec["qc_strict"] = rec.stable & (rec.t110_cuticle_um3 / rec.t110_body_um3).between(0.03, 0.40)
within = lambda a, b: (a > 0) & (b > 0) & ((a / b.where(b > 0)).between(1 / 1.10, 1.10))
use110 = within(rec.t110_body_um3, rec.t130_body_um3); use145 = ~use110 & within(rec.t145_body_um3, rec.t130_body_um3)
for q in ("body", "cuticle"):
    rec[f"ad_{q}_um3"] = np.where(use110, rec[f"t110_{q}_um3"], np.where(use145, rec[f"t145_{q}_um3"], np.nan))
rec["qc_pass"] = (use110 | use145) & (rec.ad_cuticle_um3 / rec.ad_body_um3).between(0.03, 0.40)
assert int(rec.qc_pass.sum()) == 141 and int(rec.qc_strict.sum()) == 88


def species_soft(mode):
    s = spec[spec.main_ok & spec.log10_body.notna()].copy()
    if mode == "adaptive":
        g = rec[rec.qc_pass]; ours = s.key.map(g.ad_body_um3) - s.key.map(g.ad_cuticle_um3)
    else:
        g = rec[rec.qc_strict]; ours = s.key.map(g.t110_body_um3) - s.key.map(g.t110_cuticle_um3)
    s1 = s.s1_body_um3 - s.s1_cuticle_um3
    soft = s1.where(s1.notna(), ours).where(~s.excl_body_damage.astype(bool))
    s["src"] = np.where(s1.notna(), "S1", np.where(ours.notna(), "ours", "none"))
    s["log10_soft_x"] = np.log10(soft.where(soft > 0) / 1e9)
    s = s[s.stained.values == s.species_key.map(df.set_index("species_key").stain).values]
    return s.groupby("species_key").agg(log10_soft_x=("log10_soft_x", "mean"),
                                        src=("src", lambda x: ",".join(sorted(set(x) - {"none"}))))


for mode, n_expect, est_expect in (("adaptive", 235, 0.0097544), ("strict", 197, 0.0018478)):
    d = df.merge(species_soft(mode), on="species_key", how="left").dropna(subset=["log10_soft_x"]).reset_index(drop=True)
    soft = B.fit_primary(d, mcc, SESOI, size="log10_soft_x")[0]; body = B.fit_primary(d, mcc, SESOI)[0]
    assert len(d) == n_expect and abs(soft["estimate"] - est_expect) < 1e-6, (mode, len(d), soft["estimate"])  # = r3x.json
    out = {"n": len(d), "soft_ml": brief(soft), "body_ml": brief(body), "lambda_matched": []}
    for lam in (1e-7, round(body["lambda"], 4), 1.0):
        s2 = B.fit_primary(d, mcc, SESOI, size="log10_soft_x", lam=lam)[0]; b2 = B.fit_primary(d, mcc, SESOI, lam=lam)[0]
        out["lambda_matched"].append({"lambda": lam, "soft": brief(s2), "body": brief(b2)})
    res[f"R3x_{mode}"] = out
    if mode == "adaptive":
        d["cuticle_frac"] = 1 - 10 ** (d.log10_soft_x - d.log10_body)
        d[["species_key", "subfamily", "src", "log10_colony_any", "log10_body", "log10_soft_x", "cuticle_frac"]] \
            .to_csv(OUT / "species_soft_adaptive.csv", index=False)

# 6. body-volume quality sensitivity of the primary model
bf = 100 * spec.brain_um3 / spec.body_um3
keep_cols = ["species_key", "tip", "how", "log10_colony_any"]


def refit(mask):
    s_ = spec.copy(); s_["main_ok"] = s_.main_ok & mask
    d_ = B.species_means(s_).merge(df[keep_cols], on="species_key", how="inner").reset_index(drop=True)
    t_ = B.fit_primary(d_, mcc, SESOI)[0]
    return brief(t_) | {"size_slope": t_["size_slope"]}


bq = {"all_rebuilt": refit(pd.Series(True, index=spec.index))}
assert abs(bq["all_rebuilt"]["estimate"] - t0["estimate"]) < 1e-9          # rebuild reproduces the primary fit
bq["drop_brain_body_gt10pct"] = refit(~(bf > 10).fillna(False))
bq["drop_brain_body_gt5pct"] = refit(~(bf > 5).fillna(False))
bq["drop_all_autoflagged"] = refit(~(spec.flag_touches_edge | spec.flag_fragmented | spec.flag_brain_body_frac))
bq["S1_bodies_only"] = refit(spec.body_source.eq("S1"))
bq["own_v8_bodies_only"] = refit(spec.body_source.eq("v6"))       # 'v6' is the label of the v8 body run (see INDEX sec. 7)
m_ = spec[spec.main_ok & spec.log10_body.notna()].assign(bf=lambda x: 100 * x.brain_um3 / x.body_um3)
bq["brain_frac_by_source"] = {src: {"n": int(len(g)), "median": float(g.bf.median()), "p99": float(g.bf.quantile(.99)),
                                    "max": float(g.bf.max()), "n_gt5": int((g.bf > 5).sum()), "n_gt10": int((g.bf > 10).sum())}
                              for src, g in m_.groupby("body_source")}
bq["specimens_gt10pct"] = m_[m_.bf > 10][["key", "species_key", "body_source", "bf"]].round(2).to_dict("records")
bq["corr_colony_body_primary"] = float(np.corrcoef(df.log10_colony_any, df.log10_body)[0, 1])
res["body_quality"] = bq

# 7. like-for-like: same-instrument body and cuticle on the same specimens
s7 = spec[spec.main_ok & spec.log10_body.notna()].copy()
g7 = rec[rec.qc_pass]
ob, oc = s7.key.map(g7.ad_body_um3), s7.key.map(g7.ad_cuticle_um3)
has_s1 = s7.s1_body_um3.notna() & s7.s1_cuticle_um3.notna()
s7["bodyx"] = np.where(has_s1, s7.s1_body_um3, ob); s7["softx"] = s7.bodyx - np.where(has_s1, s7.s1_cuticle_um3, oc)
s7.loc[s7.excl_body_damage.astype(bool), ["bodyx", "softx"]] = np.nan
s7["src"] = np.where(has_s1, "S1", np.where(ob.notna(), "ours", "none"))
s7 = s7[(s7.stained.values == s7.species_key.map(df.set_index("species_key").stain).values) & (s7.softx > 0)]
sp7 = s7.groupby("species_key").agg(log10_body_x=("bodyx", lambda v: np.log10(v / 1e9).mean()),
                                    log10_soft_x=("softx", lambda v: np.log10(v / 1e9).mean()),
                                    src=("src", lambda x: ",".join(sorted(set(x)))))
d7 = df.merge(sp7, on="species_key").reset_index(drop=True)
assert len(d7) == 235
lfl = {}
for name, sub in (("all_235", d7), ("S1_154", d7[d7.src == "S1"]), ("ours_81", d7[d7.src == "ours"])):
    sub = sub.reset_index(drop=True)
    lfl[name] = {"n": int(len(sub)),
                 "body_same_instrument": brief(B.fit_primary(sub, mcc, SESOI, size="log10_body_x")[0]),
                 "soft_same_specimens": brief(B.fit_primary(sub, mcc, SESOI, size="log10_soft_x")[0]),
                 "body_primary_table": brief(B.fit_primary(sub, mcc, SESOI)[0]),
                 "corr_colony_log_soft_over_body": float(np.corrcoef(sub.log10_colony_any, sub.log10_soft_x - sub.log10_body_x)[0, 1]),
                 "median_cuticle_fraction": float((1 - 10 ** (sub.log10_soft_x - sub.log10_body_x)).median())}
    for lam in (1e-7, 1.0):
        lfl[name][f"lambda_{lam:g}"] = {k: brief(B.fit_primary(sub, mcc, SESOI, size=sz, lam=lam)[0])
                                         for k, sz in (("body_same_instrument", "log10_body_x"), ("soft_same_specimens", "log10_soft_x"))}
res["like_for_like"] = lfl
sa = pd.read_csv(OUT / "species_soft_adaptive.csv").rename(columns={"cuticle_frac": "cuticle_frac_mixed"})
sa = sa.merge(d7[["species_key", "log10_body_x"]], on="species_key", how="left")
sa["cuticle_frac_same"] = 1 - 10 ** (sa.log10_soft_x - sa.log10_body_x)
sa.to_csv(OUT / "species_soft_adaptive.csv", index=False)

(OUT / "supp_checks.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("R1_fbd_200", "R12_owners_meanVCV_200", "S1_154_body", "body_quality", "like_for_like")}, indent=1, default=float)[:9000])
