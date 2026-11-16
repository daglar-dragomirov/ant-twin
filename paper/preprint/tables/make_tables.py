"""Table 1 and supplementary tables for the ANT-TWIN preprint draft (CSV + one Markdown file).

Inputs are the same local copies as figures/make_figures.py, plus supp_checks.json from supp_checks.py.
Run:  uv run --no-project --python 3.13 --with pandas,numpy,scipy python make_tables.py
Output (next to this script): table1_robustness.csv, tableS1..S8 *.csv, tables.md
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path("C:/projects/dsc-biology/research/ant-twin")
OUT = pathlib.Path(__file__).resolve().parent
WORK = ROOT / "work"; RO = WORK / "robustness/run_outputs"
SESOI = 0.02
prim = json.loads((RO / "primary.json").read_text()); summ = json.loads((RO / "summary.json").read_text())
haller = json.loads((RO / "haller.json").read_text())
r8 = json.loads((WORK / "robustness/r8.json").read_text()); r2 = json.loads((WORK / "robustness/r2.json").read_text())
r3x = json.loads((WORK / "robustness/r3x.json").read_text()); supp = json.loads((OUT / "supp_checks.json").read_text())
brms = {m: pd.read_csv(WORK / f"robustness/me/brms_summary_m_{m}_iter4000.csv").set_index("variable") for m in ("main", "tk")}
brms2000 = pd.read_csv(WORK / "robustness/me/brms_summary_m_main_iter2000.csv").set_index("variable")
mcmc = pd.read_csv(WORK / "robustness/me/mcmcglmm_summary_m_main.csv").set_index("term")
LAB = {"support_H2": "support H2", "undecided": "undecided", "equivalence_H0": "equivalence (H0)",
       "support_H1": "support H1", "small_but_nonzero": "small but non-zero"}


def classify(est, lo95, hi95, lo90, hi90):
    excl0 = lo95 > 0 or hi95 < 0
    if excl0 and -SESOI < lo95 and hi95 < SESOI: return "small_but_nonzero"
    if hi95 < 0: return "support_H1"
    if lo95 > 0: return "support_H2"
    if -SESOI < lo90 and hi90 < SESOI: return "equivalence_H0"
    return "undecided"


f3 = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}"
ci = lambda lo, hi: "" if lo is None or (isinstance(lo, float) and np.isnan(lo)) else f"[{lo:+.3f}; {hi:+.3f}]"
pfmt = lambda p: "" if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.0001" if p < 1e-4 else f"{p:.4f}")
rows = []


def add(id_, check, prereg, n, est, lo, hi, lo90, hi90, p=None, lam=None, outcome=None, source=""):
    o = outcome or classify(est, lo, hi, lo90 if lo90 is not None else -1, hi90 if hi90 is not None else 1)
    rows.append({"id": id_, "check": check, "preregistered": prereg, "n": n, "beta2": f3(est), "ci95": ci(lo, hi),
                 "ci90": ci(lo90, hi90), "p": pfmt(p), "lambda": "" if lam is None else f"{lam:.2f}", "outcome": LAB[o],
                 "source": source})


def fit(id_, check, r, prereg="yes", source="primary.json", n=None):
    n_ = r.get("n") or n
    lo90, hi90 = r["ci90"] if "ci90" in r else (r["estimate"] - stats.t.ppf(0.95, n_ - 6) * r["se"],
                                                r["estimate"] + stats.t.ppf(0.95, n_ - 6) * r["se"])
    add(id_, check, prereg, n_, r["estimate"], r["ci_low"], r["ci_high"], lo90, hi90, r.get("p"), r.get("lambda"),
        r.get("outcome"), source)


def pool(id_, check, s, n, prereg="yes", source="primary.json"):
    add(id_, check + f" (median; pooled interval; {s['n_trees']} trees/resamples; share p<0.05 = {s['share_p_lt_0.05']:.2f})",
        prereg, n, s["estimate_median"], s["pooled_ci_low"], s["pooled_ci_high"], s["pooled_ci90_low"], s["pooled_ci90_high"],
        None, s.get("lambda_median"), s.get("outcome"), source)


R = prim["robustness"]
fit("P", "Primary: PGLS-λ, Economo 2018 MCC tree", prim["mcc"])
pool("P-post", "Primary over 100 NC-stem posterior trees", prim["posterior"], 270)
b = brms["main"].loc["b_log10_colony"]
add("ME", "Measurement-error model, brms (m = 0.0311; specimen level, 442 specimens; posterior mean, CrI)", "yes", 270,
    b["mean"], b["2.5%"], b["97.5%"], b["5%"], b["95%"], source="me/brms_summary_m_main_iter4000.csv")
bt = brms["tk"].loc["b_log10_colony"]
add("ME-tk", "Measurement-error model, brms, m = 0.0127 (sensitivity)", "yes", 270, bt["mean"], bt["2.5%"], bt["97.5%"],
    bt["5%"], bt["95%"], source="me/brms_summary_m_tk_iter4000.csv")
mc = mcmc.loc["log10_colony"]
add("ME-mcmc", "Measurement-error model, MCMCglmm cross-check (posterior mean, 95% CrI)", "yes (fallback)", 270,
    mc["mean"], mc["q025"], mc["q975"], None, None, source="me/mcmcglmm_summary_m_main.csv")
pool("R1", "R1 FBD posterior trees as run (3 crown + 3 stem)", R["R1"]["result"], 270)
pool("R1-200", "R1 all 200 FBD posterior trees (draft check)", supp["R1_fbd_200"], 270, prereg="draft check",
     source="preprint/tables/supp_checks.json")
k = [k for k in r2 if k.startswith("129-158Ma_independent") and "272" in k and "| 100 resamples" in k][0]
pool("R2", "R2 Borowiec 2025 tree, one random species per genus, 117 genera", r2[k]["summary"], 117, source="r2.json")
fit("R3", "R3 soft tissue (S1 body − cuticle) as size measure", R["R3"]["result"]["soft_tissue"])
for sf, r in R["R4"]["result"].items():
    fit("R4", f"R4 without {sf}", r)
fit("R5", "R5 unstained species only", R["R5"]["result"])
fit("R6", "R6 all workers incl. major/media/intercaste", R["R6"]["result"])
fit("R7", "R7 without cf./aff./nr. names", R["R7"]["result"])
fit("R8", "R8 without supercolonial species and social parasites, MCC", r8["mcc"], source="r8.json")
pool("R8-post", "R8 over 100 posterior trees", r8["posterior"], 251, source="r8.json")
fit("R9", "R9 colony size from Bell-Roberts 2024 only", R["R9"]["result"]["log10_colony_br"])
fit("R9", "R9 maximum instead of mean colony size", R["R9"]["result"]["log10_colony_max_owners"])
fit("R10", "R10 genus-level colony size (approximation, see deviations)", R["R10"]["result"])
fit("R11", "R11 without genus-placed tips", R["R11"]["result"])
fit("R12", "R12 λ fixed at 1 (Brownian motion)", R["R12"]["result"]["lambda_1"])
fit("R12", "R12 owners' method: mean VCV of 6 FBD trees / 200, fixed", R["R12"]["result"]["owners_meanVCV_div200"])
fit("R12-200", "R12 owners' method on all 200 FBD trees (draft check)", supp["R12_owners_meanVCV_200"], prereg="draft check",
    source="preprint/tables/supp_checks.json")
fit("R13", "R13 badly preserved specimens returned + collection year covariate", R["R13"]["result"])
fit("R14", "R14 OLS without phylogeny", R["R14"]["result"])
f = r3x["fits"]
fit("R3x", "R3 extension: soft tissue, S1 + own measurements, adaptive QC (main)", f["R3x_adaptive"], "no (declared after unblinding)", "r3x.json")
pool("R3x-post", "R3 extension, adaptive QC, 100 posterior trees", {**f["R3x_chosen_posterior_trees"], "n_trees": 100,
     "pooled_ci90_low": np.nan, "pooled_ci90_high": np.nan, "lambda_median": None}, 235, "no", "r3x.json")
fit("R3x-body", "Body denominator, same 235 species", f["R3x_adaptive_body_same_species"], "no", "r3x.json")
fit("R3x-strict", "R3 extension: soft tissue, strict QC (sensitivity)", f["R3x_t110"], "no", "r3x.json")
fit("R3x-strict-body", "Body denominator, same 197 species", f["R3x_t110_body_same_species"], "no", "r3x.json")
fit("R3-body154", "Body denominator, same 154 S1 species as R3 (exploratory)", supp["S1_154_body"], "no",
    "preprint/tables/supp_checks.json (= primary_v1.md)")
lf, bq = supp["like_for_like"], supp["body_quality"]
fit("LFL-body", "Body from the same instrument and specimens as soft tissue (S1 or owners' program), 235 species", lf["all_235"]["body_same_instrument"],
    "no (draft check)", "preprint/tables/supp_checks.json")
fit("LFL-body-S1", "Same, S1 species only (owners' body volumes)", lf["S1_154"]["body_same_instrument"], "no (draft check)", "preprint/tables/supp_checks.json")
for k_, lab in (("drop_brain_body_gt10pct", "Primary model, specimens with brain/body > 10% removed"),
                ("drop_brain_body_gt5pct", "Primary model, specimens with brain/body > 5% removed"),
                ("drop_all_autoflagged", "Primary model, all autoflagged specimens removed"),
                ("S1_bodies_only", "Primary model, owners' S1 body volumes only"),
                ("own_v8_bodies_only", "Primary model, own v8 body volumes only")):
    fit("BQ", lab, bq[k_], "no (draft check)", "preprint/tables/supp_checks.json")
t1 = pd.DataFrame(rows)
t1.to_csv(OUT / "table1_robustness.csv", index=False)

# ---------------------------------------------------------------- S1 funnel
fu, cn = summ["funnel"], summ["counts"]
s1 = pd.DataFrame([
    ("Target set: colony workers of species with colony size (brain_table_v1.md)", "818 specimens / 352 species"),
    ("Workers with an agent-approved brain mask (frozen table)", f"{fu['workers_with_brain_record']} specimens / 325 species"),
    ("Excluded for damage (head/anterior truncation etc.)", fu["damage_excluded"]),
    ("Brain OK", fu["brain_ok"]),
    ("Excluded major / media / intercaste (main analysis)", fu["major_media_intercaste"]),
    ("Main-analysis workers", fu["main_ok"]),
    ("Main-analysis workers with body volume", f"{fu['main_ok_with_body']} (S1 {fu['body_source_main_ok']['S1']}, own v8 {fu['body_source_main_ok']['v6']})"),
    ("Species with QC brain", cn["species_with_qc_brain"]),
    ("Species with QC brain and body", cn["species_with_qc_brain_and_body"]),
    ("Species with brain, body, colony size, in the Economo tree (primary n)", cn["species_brain_body_colony_in_tree"]),
    ("  of which unstained", cn["species_brain_body_colony_in_tree_unstained"]),
], columns=["step", "count"])
s1.to_csv(OUT / "tableS1_funnel.csv", index=False)

# ---------------------------------------------------------------- S2 segmentation accuracy
seg = []
for k_ in range(5):
    s = json.loads((WORK / f"seg_v3/fold_{k_}_summary.json").read_text())
    for src, lab in (("TK", "Toulkeridou 2023, human labels"), ("AS", "AntScan, gate-1 reference drafts"),
                     ("AS2", "AntScan, agent-approved reference masks")):
        seg.append({"fold": k_, "reference": lab, "n": s[src]["n"], "dice_median": round(s[src]["dice_median"], 4),
                    "abs_vol_err_median_pct": round(100 * s[src]["abs_vol_err_median"], 2)})
s2 = pd.DataFrame(seg)
s2.to_csv(OUT / "tableS2_segmentation.csv", index=False)

# ---------------------------------------------------------------- S3 coefficients
s3 = pd.concat([pd.DataFrame(prim["coef_mcc"]).assign(model=f"Primary (MCC, λ = {prim['mcc']['lambda']:.3f}, n = 270)"),
                pd.DataFrame(haller["coef"]).assign(model=f"Gate 3 / Haller (no colony, λ = {haller['lambda']:.3f}, n = 270)")])
s3 = s3[["model", "term", "estimate", "se", "t", "p", "ci_low", "ci_high"]].round(5)
s3.to_csv(OUT / "tableS3_coefficients.csv", index=False)

# ---------------------------------------------------------------- S4 R2 variants
s4 = []
for k_, v in r2.items():
    s = v["summary"]
    s4.append({"tree / species set / resamples": k_, "genera": s["n_species_min"], "median_beta2": round(s["estimate_median"], 4),
               "q025_q975_of_resamples": f"[{s['estimate_q025']:.3f}; {s['estimate_q975']:.3f}]",
               "pooled_ci95": f"[{s['pooled_ci_low']:+.3f}; {s['pooled_ci_high']:+.3f}]",
               "share_H2": s["outcome_counts"].get("support_H2", 0) / s["n_trees"],
               "share_negative_estimates": 1 - s["share_estimate_pos"], "lambda_median": round(s["lambda_median"], 2)})
g = json.loads((WORK / "robustness/r2_economo_genus.json").read_text())["summary"]
s4.append({"tree / species set / resamples": "EXPLORATORY control: Economo MCC tree, same 1-sp./genus draws, 270 primary species | 100 resamples",
           "genera": g["n_species_min"], "median_beta2": round(g["estimate_median"], 4),
           "q025_q975_of_resamples": f"[{g['estimate_q025']:.3f}; {g['estimate_q975']:.3f}]",
           "pooled_ci95": f"[{g['pooled_ci_low']:+.3f}; {g['pooled_ci_high']:+.3f}]",
           "share_H2": g["outcome_counts"].get("support_H2", 0) / g["n_trees"], "share_negative_estimates": 1 - g["share_estimate_pos"],
           "lambda_median": round(g["lambda_median"], 2)})
s4 = pd.DataFrame(s4); s4.to_csv(OUT / "tableS4_r2_borowiec.csv", index=False)

# ---------------------------------------------------------------- S5 R3 extension: QC and validation
q = r3x["qc"]; va = r3x["validation_adaptive"]; vt = r3x["validation"]
s5 = [{"item": "Specimens measured with the owners' program", "value": q["n"]},
      {"item": "  targets (primary species without S1) / S1 validation", "value": "141 / 44"},
      {"item": "Pass adaptive QC (threshold 110 / 145)", "value": f"{q['pass']} ({q['thr_used']['t110']} / {q['thr_used']['t145']})"},
      {"item": "Pass strict QC (threshold 110, body stable across thresholds ≤1.15)", "value": q["pass_strict"]},
      {"item": "Adaptive-QC failures by voxel size 6.11 / 2.44 / 1.22 µm",
       "value": f"{q['fail_by_voxel']['6.11']}/{q['n_by_voxel']['6.11']}, {q['fail_by_voxel']['2.44']}/{q['n_by_voxel']['2.44']}, {q['fail_by_voxel']['1.22']}/{q['n_by_voxel']['1.22']}"},
      {"item": f"Validation vs S1, adaptive (n = {va['n']} of {r3x['validation_n_all']}): median ratio body / cuticle / soft",
       "value": f"{va['body_median_ratio']:.3f} / {va['cuticle_median_ratio']:.3f} / {va['soft_median_ratio']:.3f}"},
      {"item": "  r(log body), r(log soft)", "value": f"{va['r_log_body']:.3f}, {va['r_log_soft']:.3f}"},
      {"item": "  excluded validation specimens", "value": ", ".join(va["excluded"])}]
for t in ("t110", "t130", "t145"):
    v = vt[t]
    s5.append({"item": f"Validation, threshold {t[1:]}, stable specimens (n = {v['n']}): median ratio body / cuticle / soft; r(log soft)",
               "value": f"{v['body_median_ratio']:.3f} / {v['cuticle_median_ratio']:.3f} / {v['soft_median_ratio']:.3f}; {v['r_log_soft']:.3f}"})
gp = pd.read_csv(WORK / "robustness/gap1_vs_gap3.csv")
for col, lab in (("body_g3_g1", "body"), ("soft_g3_g1", "soft tissue")):
    x = gp[gp.t == "t110"][col].dropna()
    s5.append({"item": f"Gap-3 / gap-1 ratio, {lab}, threshold 110 (n = {len(x)} specimens): median (range)",
               "value": f"{x.median():.4f} ({x.min():.3f}–{x.max():.3f})"})
s5 = pd.DataFrame(s5); s5.to_csv(OUT / "tableS5_r3x_qc_validation.csv", index=False)

# ---------------------------------------------------------------- S6 R3 extension fits + lambda-matched
s6 = []
for key, lab in (("R3_prereg_S1_only", "Soft tissue, S1 only (preregistered R3)"), ("R3x_adaptive", "Soft tissue, S1 + own, adaptive QC"),
                 ("R3x_adaptive_body_same_species", "Body, same species (adaptive set)"), ("R3x_t110", "Soft tissue, strict QC (t110)"),
                 ("R3x_t110_body_same_species", "Body, same species (strict set)"), ("R3x_t130", "Soft tissue, strict QC, threshold 130"),
                 ("R3x_t145", "Soft tissue, strict QC, threshold 145")):
    r = f[key]; lo90, hi90 = r["estimate"] - stats.t.ppf(0.95, r["n"] - 6) * r["se"], r["estimate"] + stats.t.ppf(0.95, r["n"] - 6) * r["se"]
    s6.append({"fit": lab, "lambda": "ML", "n": r["n"], "lambda_hat": round(r["lambda"], 3), "beta2": f3(r["estimate"]),
               "ci95": ci(r["ci_low"], r["ci_high"]), "ci90": ci(lo90, hi90), "p": pfmt(r["p"]), "outcome": LAB[r["outcome"]], "source": "r3x.json"})
s6.append({"fit": "Body, same 154 S1 species", "lambda": "ML", "n": supp["S1_154_body"]["n"], "lambda_hat": round(supp["S1_154_body"]["lambda"], 3),
           "beta2": f3(supp["S1_154_body"]["estimate"]), "ci95": ci(supp["S1_154_body"]["ci_low"], supp["S1_154_body"]["ci_high"]),
           "ci90": ci(*supp["S1_154_body"]["ci90"]), "p": pfmt(supp["S1_154_body"]["p"]), "outcome": LAB[supp["S1_154_body"]["outcome"]],
           "source": "supp_checks.json"})
for mode in ("adaptive", "strict"):
    for m in supp[f"R3x_{mode}"]["lambda_matched"]:
        for den in ("soft", "body"):
            r = m[den]
            s6.append({"fit": f"{'Soft tissue' if den == 'soft' else 'Body'}, {mode} set, λ fixed (draft check)",
                       "lambda": f"{m['lambda']:.3g}", "n": r["n"], "lambda_hat": "", "beta2": f3(r["estimate"]),
                       "ci95": ci(r["ci_low"], r["ci_high"]), "ci90": ci(*r["ci90"]), "p": pfmt(r["p"]), "outcome": LAB[r["outcome"]],
                       "source": "supp_checks.json"})
for name, lab in (("all_235", "235 species (S1 + own)"), ("S1_154", "154 S1 species"), ("ours_81", "81 own-measured species")):
    L = lf[name]
    for key, den in (("body_same_instrument", "Body, same instrument/specimens"), ("soft_same_specimens", "Soft tissue, same specimens"),
                     ("body_primary_table", "Body, primary table")):
        r = L[key]
        s6.append({"fit": f"Like-for-like, {lab}: {den} (draft check)", "lambda": "ML", "n": r["n"], "lambda_hat": round(r["lambda"], 3),
                   "beta2": f3(r["estimate"]), "ci95": ci(r["ci_low"], r["ci_high"]), "ci90": ci(*r["ci90"]), "p": pfmt(r["p"]),
                   "outcome": LAB[r["outcome"]], "source": "supp_checks.json"})
    for lamk in ("lambda_1e-07", "lambda_1"):
        for key, den in (("body_same_instrument", "Body, same instrument"), ("soft_same_specimens", "Soft tissue")):
            r = L[lamk][key]
            s6.append({"fit": f"Like-for-like, {lab}: {den}, λ fixed (draft check)", "lambda": lamk.split("_")[1], "n": r["n"], "lambda_hat": "",
                       "beta2": f3(r["estimate"]), "ci95": ci(r["ci_low"], r["ci_high"]), "ci90": ci(*r["ci90"]), "p": pfmt(r["p"]),
                       "outcome": LAB[r["outcome"]], "source": "supp_checks.json"})
s6 = pd.DataFrame(s6); s6.to_csv(OUT / "tableS6_r3x_fits.csv", index=False)
bqf = pd.DataFrame([{"source": k_, **v} for k_, v in bq["brain_frac_by_source"].items()]).round(2)
bqf["source"] = bqf.source.replace({"v6": "own v8 body mask", "S1": "owners' S1"})
bqf.to_csv(OUT / "tableS9_brain_fraction_by_body_source.csv", index=False)
sp10 = pd.DataFrame(bq["specimens_gt10pct"])

# ---------------------------------------------------------------- S7 measurement-error models
s7 = []
for lab, d in (("brms, m = 0.0311, 4 × 4000 it.", brms["main"]), ("brms, m = 0.0127, 4 × 4000 it.", brms["tk"]),
               ("brms, m = 0.0311, 4 × 2000 it. (prereg. length)", brms2000)):
    for v in ("b_log10_colony", "b_log10_body", "b_iodine", "b_vox122", "b_vox611", "sd_phylo__Intercept", "sd_species__Intercept", "sigma"):
        r = d.loc[v]
        s7.append({"model": lab, "parameter": v, "mean": round(r["mean"], 4), "ci95": f"[{r['2.5%']:.4f}; {r['97.5%']:.4f}]",
                   "rhat": round(r["rhat"], 4), "ess_bulk": int(r["ess_bulk"]), "ess_tail": int(r["ess_tail"])})
for v, r in mcmc.iterrows():
    s7.append({"model": "MCMCglmm, m = 0.0311", "parameter": v, "mean": round(r["mean"], 4), "ci95": f"[{r['q025']:.4f}; {r['q975']:.4f}]",
               "rhat": f"PSRF {r['psrf']:.4f}", "ess_bulk": int(r["ess"]), "ess_tail": ""})
s7 = pd.DataFrame(s7); s7.to_csv(OUT / "tableS7_measurement_error.csv", index=False)

# ---------------------------------------------------------------- S8 R8 dropped species
s8 = pd.DataFrame(r8["dropped"])[["species_key", "subfamily", "supercolonial", "social_parasite", "log10_colony_any"]]
s8["species"] = s8.species_key.str.replace("_", " ").str.capitalize(); s8 = s8[["species", "subfamily", "supercolonial", "social_parasite", "log10_colony_any"]].round(2)
s8.to_csv(OUT / "tableS8_r8_dropped.csv", index=False)

# ---------------------------------------------------------------- markdown
def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


parts = ["# ANT-TWIN preprint — tables (generated by `make_tables.py`; do not edit by hand)\n",
         "Outcome labels follow the preregistered decision rules (prereg sec. 7): support H2 = 95% CI > 0; support H1 = 95% CI < 0; "
         "equivalence = 90% CI inside (−0.02; 0.02); otherwise undecided. 90% intervals for single fits are t-intervals with n − 6 df. "
         "For tree sets and resamples, β₂ is the median and the interval is the pooled t-mixture interval. "
         "'Draft check' rows were computed for this draft by `supp_checks.py` and are not in the committed results files.\n",
         "## Table 1. β₂ (log10 colony size) in the primary analysis and all robustness checks\n", md(t1.drop(columns=["source"])),
         "\nSources: " + "; ".join(sorted(set(t1.source))) + ".\n",
         "## Table S1. Sample funnel\n", md(s1),
         "\n## Table S2. Segmentation accuracy of seg v3 on held-out folds (species-level folds within each source)\n", md(s2),
         "\n## Table S3. Coefficients of the primary model and the Haller's-rule gate\n", md(s3),
         "\n## Table S4. R2: Borowiec et al. 2025 tree, all dating schemes and species sets\n", md(s4),
         "\n## Table S5. R3 extension: specimen QC and validation against S1\n", md(s5),
         "\n## Table S6. R3 extension: fits with soft-tissue and body denominators, including λ-matched comparisons\n", md(s6),
         "\n## Table S7. Measurement-error models (specimen level)\n", md(s7),
         "\n## Table S8. Species dropped in R8 (Bell-Roberts et al. 2024, Supplementary Table 21)\n", md(s8),
         "\n## Table S9. Brain/body volume ratio (%) of main-analysis workers by body-volume source (draft diagnostic)\n", md(bqf),
         "\nWorkers with brain/body > 10%:\n", md(sp10)]
(OUT / "tables.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
print(t1[["id", "n", "beta2", "ci95", "outcome"]].to_string())
