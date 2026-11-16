"""Stage brain_analysis: brain table -> Haller's-rule positive control -> preregistered primary PGLS (guarded) -> robustness.

Preregistration: dsc-biology research/ant-twin/prereg/prereg_draft.md (v0.1). Section numbers below refer to it.

1. Specimen table (sec. 4): AntScan workers; brain volume from seg-predict runs (newest reviewed record wins per key),
   vision-agent QC filter (configurable), automatic flags (recorded), damage exclusions from `problems`,
   body volume rule (owners' S1 > our v6 if guarded_slices < guard_max > missing; source recorded), soft tissue
   (S1 body - S1 cuticle), stain, scan step, folder.
2. Species table (sec. 4, "species mean"): log10 means (mm^3) of QC-passed workers of one preparation (unstained if any,
   else iodine), modal scan-step class (8.2 -> 6.11), n specimens, measurement SE of log10 brain
   (SD/sqrt(n) if n > 1, else log10(1 + imputed_rel_error)).
3. Gate 3 (sec. 5): PGLS-lambda log10 brain ~ log10 body + stain + step class on the MCC tree; pass/fail thresholds from cfg.
4. Primary model (sec. 6-7): log10 brain ~ log10 body + log10 colony + iodine + step1.22 + step6.11, MCC + posterior trees,
   decision rules incl. TOST. Executed on real brains ONLY if cfg["run_primary"] is true. The dry run exercises the same
   code on synthetic brains simulated on the real tree / species set / colony sizes under known slopes.
5. Robustness runner (sec. 8): one function per check R1-R14; implemented ones are run on synthetic data in the dry run.

Blinding (sec. 2): while cfg["blind"] is true, colony values are never written next to real brain values (output tables
carry only a has_colony flag) and no model containing colony is fitted to real brains.
"""
import io, re, json, time, hashlib, pathlib
import numpy as np, pandas as pd
from scipy import stats
import r2
from analysis import pgls
from stages.cuticle_replication import trees_dir, sha

UM3_PER_MM3 = 1e9
VOX_CLASSES = {1.22: "1.22", 2.44: "2.44", 6.11: "6.11", 8.2: "6.11", 8.20: "6.11"}


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def spec_key(folder, vial):
    return f"{folder}__{vial_code(vial)}"


def log(*a):
    print("[brain_analysis]", *a, flush=True)


# ============================================================== inputs

def r2_get(c, key, dest):
    dest = pathlib.Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    c.download_file(r2.BUCKET, key, str(dest)); return dest


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def resolve_body_run(c, cfg):
    """Explicit cfg['body_run'] or the newest run matching cfg['body_run_match'] with the most rows (status ok)."""
    if cfg.get("body_run"):
        return cfg["body_run"]
    keys = r2.list_keys(c, r2.BUCKET, "ant-twin/runs/")
    cands = sorted({k.split("/")[2] for k in keys if cfg["body_run_match"] in k and k.endswith("/outputs/body.jsonl")})
    best, best_n = None, -1
    for run in cands:  # sorted by timestamp prefix -> ties go to the newest
        size = keys[f"ant-twin/runs/{run}/outputs/body.jsonl"]
        if size >= best_n:
            best, best_n = run, size
    return best


def aggregate_reviews(recs):
    """Several reviews of one key -> one verdict: majority mask (ties -> worse), box_on_head true only if all true,
    brain_fully_in_box false if any false, else true if all true, else 'unsure'."""
    order = {"bad": 0, "fix": 1, "good": 2}
    out = {}
    for k, rs in pd.DataFrame(recs).groupby("key"):
        masks = rs["mask"].tolist()
        cnt = pd.Series(masks).value_counts()
        top = cnt[cnt == cnt.max()].index.tolist()
        mask = min(top, key=lambda m: order.get(m, -1))
        boh = rs["box_on_head"].tolist(); fib = rs["brain_fully_in_box"].tolist()
        out[k] = {"mask": mask, "box_on_head": True if all(b is True for b in boh) else (False if any(b is False for b in boh) else "unsure"),
                  "brain_fully_in_box": False if any(b is False for b in fib) else (True if all(b is True for b in fib) else "unsure"),
                  "est_vol_error_pct": float(pd.to_numeric(rs.get("est_vol_error_pct"), errors="coerce").median()) if "est_vol_error_pct" in rs else np.nan,
                  "n_reviews": int(len(rs))}
    return out


def load_brains(c, cfg, inp, sources_log):
    """All brain records from the configured sources; per key the newest source whose reviews cover the key wins,
    else the newest record (it then fails QC as 'no_review')."""
    recs = []
    for si, src in enumerate(cfg["brain_sources"]):
        k = f"ant-twin/runs/{src['brains_run']}/outputs/brains.jsonl"
        p = r2_get(c, k, inp / "brains" / f"{si}_brains.jsonl"); sources_log[f"brains_{si}"] = {"r2": k, "sha256": sha(p)}
        rv = []
        for j, rk in enumerate(src.get("reviews", [])):
            q = r2_get(c, rk, inp / "reviews" / f"{si}_{j}.json"); sources_log[f"review_{si}_{j}"] = {"r2": rk, "sha256": sha(q)}
            rv += json.loads(q.read_text(encoding="utf-8"))
        rvmap = aggregate_reviews(rv) if rv else {}
        for b in read_jsonl(p):
            r = {"key": b["key"], "brain_um3": b.get("brain_um3"), "brain_vox": b.get("brain_vox"), "brain_voxel_um": b.get("voxel_um"),
                 "touches_edge": b.get("touches_edge"), "n_components": b.get("n_components"), "largest_frac": b.get("largest_frac"),
                 "box_frac": b.get("box_frac"), "fold_cv": b.get("fold_cv"), "brains_run": src["brains_run"],
                 "mask_version": src.get("mask_version", f"src{si}"), "run_ts": src["brains_run"][:15]}
            rev = rvmap.get(b["key"])
            r.update({"reviewed": rev is not None, **({f"rev_{kk}": vv for kk, vv in rev.items()} if rev else {})})
            recs.append(r)
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    d = d.sort_values(["reviewed", "run_ts"], ascending=[False, False]).drop_duplicates("key").reset_index(drop=True)
    return d


def problem_items(s):
    return [x.strip().lower() for x in re.split(r"[,;]", str(s)) if x.strip()]


def matches_any(items, patterns):
    return any(re.search(p, it) for it in items for p in patterns)


# ============================================================== 1. specimen table

def build_specimens(cfg, spec, brains, body, s1):
    """One row per AntScan worker with a brain record; flags and inclusion reasons are kept, nothing is dropped here."""
    w = spec[spec.caste == "worker"].copy()
    w["key"] = [spec_key(f, v) for f, v in zip(w.folder, w.vial)]
    dup = int(w.key.duplicated().sum())
    w = w.drop_duplicates("key")
    d = w.merge(brains, on="key", how="inner")
    # damage (sec. 4.3)
    items = d.problems.map(problem_items)
    d["excl_damage"] = items.map(lambda it: matches_any(it, cfg["exclude_problems"]))
    d["excl_body_damage"] = items.map(lambda it: matches_any(it, cfg["body_only_problems"]))
    d["excl_caste"] = d.caste_detail.isin(cfg["exclude_caste_detail"])
    # body volume rule
    s1w = s1[s1.caste == "worker"].drop_duplicates("specimen_code").set_index("specimen_code")
    d["s1_body_um3"] = d.specimen_code.map(s1w.body_volume)
    d["s1_cuticle_um3"] = d.specimen_code.map(s1w.cuticle_volume)
    bv = body.drop_duplicates("key").set_index("key")
    d["v6_body_um3"] = d.key.map(bv.body_um3)
    d["v6_guarded_slices"] = d.key.map(bv.guarded_slices)
    d["v6_status"] = d.key.map(bv.status)
    v6_ok = (d.v6_status == "ok") & d.v6_body_um3.notna() & (d.v6_guarded_slices < cfg["body_guard_max"])
    d["body_source"] = np.where(d.s1_body_um3.notna(), "S1", np.where(v6_ok, "v6", "missing"))
    d["body_um3"] = np.where(d.body_source == "S1", d.s1_body_um3, np.where(d.body_source == "v6", d.v6_body_um3, np.nan))
    d.loc[d.excl_body_damage, ["body_um3"]] = np.nan
    d.loc[d.excl_body_damage, "body_source"] = "missing_damage"
    d["soft_um3"] = np.where(d.s1_cuticle_um3.notna() & ~d.excl_body_damage, d.s1_body_um3 - d.s1_cuticle_um3, np.nan)
    # automatic flags (sec. 4.4), recorded; which ones exclude is cfg["qc"]["autoflag_exclude"]
    frac = d.brain_um3 / d.body_um3
    lo, hi = cfg["brain_body_frac_range"]
    d["brain_body_frac"] = frac
    d["flag_touches_edge"] = d.touches_edge.fillna(False).astype(bool)
    d["flag_fragmented"] = d.largest_frac.fillna(1.0) < cfg["largest_frac_min"]
    d["flag_fold_cv"] = d.fold_cv.fillna(0) > cfg["fold_cv_max"]  # not produced by seg_predict yet -> never set
    d["flag_brain_body_frac"] = frac.notna() & ((frac < lo) | (frac > hi))
    d["flag_empty"] = ~(d.brain_um3 > 0)
    # vision-agent QC
    q = cfg["qc"]
    rev_ok = d.reviewed & d.get("rev_mask", pd.Series(index=d.index, dtype=object)).isin(q["mask"])
    rev_ok &= d.get("rev_box_on_head", pd.Series(index=d.index, dtype=object)).isin(q["box_on_head"])
    rev_ok &= ~d.get("rev_brain_fully_in_box", pd.Series(index=d.index, dtype=object)).isin(q["brain_fully_in_box_exclude"])
    d["qc_review_pass"] = rev_ok
    af = np.zeros(len(d), bool)
    for f in q.get("autoflag_exclude", []):
        af |= d["flag_" + f].to_numpy(bool)
    d["qc_autoflag_fail"] = af
    d["vox_class"] = d.voxel_um.round(2).map(VOX_CLASSES).fillna(d.voxel_um.astype(str))
    d["brain_ok"] = d.qc_review_pass & ~d.qc_autoflag_fail & ~d.flag_empty & ~d.excl_damage & d.species_key.notna()
    d["main_ok"] = d.brain_ok & ~d.excl_caste
    d["log10_brain"] = np.log10(d.brain_um3.where(d.brain_um3 > 0) / UM3_PER_MM3)
    d["log10_body"] = np.log10(d.body_um3 / UM3_PER_MM3)
    d["log10_soft"] = np.log10(d.soft_um3.where(d.soft_um3 > 0) / UM3_PER_MM3)
    d["qualified_name"] = d.name.astype(str).str.contains(r"\b(?:cf|aff|nr)\b\.?", case=False, regex=True)
    d["year"] = pd.to_datetime(d.date_collected, errors="coerce").dt.year
    funnel = {"workers_in_specimen_table": int(len(w)), "duplicate_keys_dropped": dup,
              "workers_with_brain_record": int(len(d)), "reviewed": int(d.reviewed.sum()),
              "review_pass": int(d.qc_review_pass.sum()), "empty_mask": int(d.flag_empty.sum()),
              "damage_excluded": int(d.excl_damage.sum()), "autoflag_excluded": int(d.qc_autoflag_fail.sum()),
              "no_species_key": int(d.species_key.isna().sum()), "brain_ok": int(d.brain_ok.sum()),
              "major_media_intercaste": int((d.brain_ok & d.excl_caste).sum()), "main_ok": int(d.main_ok.sum()),
              "main_ok_with_body": int((d.main_ok & d.body_um3.notna()).sum()),
              "body_source_main_ok": d[d.main_ok].body_source.value_counts().to_dict(),
              "flags_among_main_ok": {f: int(d.loc[d.main_ok, f].sum()) for f in d.columns if f.startswith("flag_")},
              "unlisted_problems_main_ok": d[d.main_ok & d.problems.astype(str).str.strip().ne("") & ~d.excl_body_damage].problems.value_counts().to_dict()}
    return d, funnel


# ============================================================== 2. species table

def species_means(d, ok_col="main_ok", pair_brain_body=True, rel_err=0.06):
    """Species means of log10 volumes over QC-passed workers of one preparation (unstained if any, else iodine)."""
    x = d[d[ok_col]].copy()
    if pair_brain_body:
        x = x[x.log10_body.notna()]
    rows = []
    se_imp = float(np.log10(1 + rel_err))
    for sp, g in x.groupby("species_key"):
        stain = "unstained" if (g.stained == "unstained").any() else "iodine"
        g = g[g.stained == stain]
        vc = g.vox_class.value_counts()
        vox = sorted(vc[vc == vc.max()].index)[0]
        n = len(g); sd = float(g.log10_brain.std(ddof=1)) if n > 1 else np.nan
        rows.append({"species_key": sp, "subfamily": g.subfamily.iloc[0], "genus": sp.split("_")[0], "stain": stain,
                     "vox_class": vox, "n_specimens": n, "log10_brain": g.log10_brain.mean(), "log10_body": g.log10_body.mean(),
                     "log10_soft": g.log10_soft.mean() if g.log10_soft.notna().any() else np.nan,
                     "n_soft": int(g.log10_soft.notna().sum()), "sd_log10_brain": sd,
                     "se_log10_brain": sd / np.sqrt(n) if n > 1 else se_imp, "se_imputed": n == 1,
                     "body_sources": ",".join(sorted(g.body_source.unique())), "qualified_name": bool(g.qualified_name.any()),
                     "year": g.year.mean(), "mask_versions": ",".join(sorted(g.mask_version.unique()))})
    return pd.DataFrame(rows)


def attach_colony(sp, spp):
    s = spp.set_index("species_key")
    for c in ("colony_any", "colony_br", "colony_max_owners", "colony_mean_owners"):
        sp["log10_" + c] = np.log10(sp.species_key.map(s[c]))
    # R10 approximation: genus median of species-level owners' colony means available in the species table
    gmed = spp.assign(genus=spp.species_key.str.split("_").str[0]).groupby("genus").colony_mean_owners.median()
    sp["log10_colony_genus"] = np.log10(sp.genus.map(gmed))
    return sp


# ============================================================== models

def drop_constant(X, names, keep=("(Intercept)",)):
    keepcols = [i for i, n in enumerate(names) if n in keep or np.ptp(X[:, i]) > 0]
    dropped = [n for i, n in enumerate(names) if i not in keepcols]
    return X[:, keepcols], [names[i] for i in keepcols], dropped


def model_design(df, size="log10_body", colony=None, extra=()):
    """Preregistered fixed effects: size (+ colony) + iodine + step dummies (reference 2.44 um). Constant columns dropped."""
    num = [size] + ([colony] if colony else []) + list(extra)
    X, names = pgls.design(df, num, {"stain": ["unstained", "iodine"], "vox_class": ["2.44", "1.22", "6.11"]})
    return drop_constant(X, names)


def r2_ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return float(1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())))


def haller(df, tree, th):
    """Gate 3: PGLS-lambda log10 brain ~ log10 body + stain + step class on the MCC tree (no colony)."""
    X, names, dropped = model_design(df)
    y = df.log10_brain.to_numpy(float)
    out = {"n_species": int(len(df)), "terms": names, "dropped_constant_terms": dropped, "thresholds": th}
    if len(df) < len(names) + 3:
        out.update(status="insufficient_data"); return out, None
    V = tree.vcv(list(df.tip))
    res = pgls.pgls_lambda(y, X, V, names)
    t = pgls.term(res, "log10_body")
    Xb = np.column_stack([np.ones(len(y)), df.log10_body.to_numpy(float)])
    r2_full, r2_body = r2_ols(y, X), r2_ols(y, Xb)
    ols_slope = float(np.linalg.lstsq(Xb, y, rcond=None)[0][1])
    passed = {"slope_in_range": th["slope_range"][0] <= t["estimate"] <= th["slope_range"][1],
              "ci_inside": th["ci_within"][0] < t["ci_low"] and t["ci_high"] < th["ci_within"][1],
              "r2_ols_species_means": r2_full > th["r2_min"]}
    out.update(status="pass" if all(passed.values()) else "fail", slope=t["estimate"], se=t["se"], ci_low=t["ci_low"],
               ci_high=t["ci_high"], p=t["p"], df=t["df"], **{"lambda": t["lambda"]},
               lambda_ci=[t.get("lambda_ci_low"), t.get("lambda_ci_high")],
               r2_ols_same_terms=r2_full, r2_ols_body_only=r2_body, ols_slope_body_only=ols_slope, checks=passed,
               resid_sd_pgls=float(np.sqrt(res["sigma2"])), coef=res["coef"].to_dict("records"))
    return out, res


def classify(est, se, df_, sesoi, lo95=None, hi95=None, lo90=None, hi90=None):
    """Sec. 7 outcome for beta2. Optional explicit intervals (pooled over trees)."""
    q975, q95 = stats.t.ppf(0.975, df_), stats.t.ppf(0.95, df_)
    lo95 = est - q975 * se if lo95 is None else lo95; hi95 = est + q975 * se if hi95 is None else hi95
    lo90 = est - q95 * se if lo90 is None else lo90; hi90 = est + q95 * se if hi90 is None else hi90
    excl0 = lo95 > 0 or hi95 < 0
    if excl0 and -sesoi < lo95 and hi95 < sesoi:
        o = "small_but_nonzero"
    elif hi95 < 0:
        o = "support_H1"
    elif lo95 > 0:
        o = "support_H2"
    elif -sesoi < lo90 and hi90 < sesoi:
        o = "equivalence_H0"
    else:
        o = "undecided"
    return {"outcome": o, "ci95": [float(lo95), float(hi95)], "ci90": [float(lo90), float(hi90)]}


def fit_primary(df, tree, sesoi, colony="log10_colony_any", size="log10_body", lam=None, extra=(), V=None):
    """Preregistered primary model on one tree. Returns the colony term, decision and all coefficients."""
    X, names, dropped = model_design(df, size=size, colony=colony, extra=extra)
    y = df.log10_brain.to_numpy(float)
    V = tree.vcv(list(df.tip)) if V is None else V
    res = pgls.pgls_lambda(y, X, V, names, lam=lam)
    t = pgls.term(res, colony)
    t.update(classify(t["estimate"], t["se"], t["df"], sesoi), dropped_constant_terms=dropped, terms=names,
             size_slope=float(res["coef"].set_index("term").loc[size, "estimate"]))
    return t, res


def pooled(per_tree, sesoi, draws=2000, seed=1):
    s = pgls.summarize_over_trees(per_tree, draws=draws, seed=seed)
    rng = np.random.default_rng(seed)
    mix = np.concatenate([r.estimate + r.se * rng.standard_t(r.df, draws) for r in per_tree.itertuples()])
    s["pooled_ci90_low"], s["pooled_ci90_high"] = float(np.quantile(mix, 0.05)), float(np.quantile(mix, 0.95))
    s.update(classify(s["estimate_median"], np.nan, 1, sesoi, s["pooled_ci_low"], s["pooled_ci_high"],
                      s["pooled_ci90_low"], s["pooled_ci90_high"]))
    return s


def primary_over_trees(df, trees, sesoi, colony="log10_colony_any", lam=None):
    rows = []
    for i, t in enumerate(trees):
        keep = df.tip.isin(set(t.tip_index)).to_numpy()
        r, _ = fit_primary(df[keep].reset_index(drop=True), t, sesoi, colony=colony, lam=lam)
        rows.append({k: r[k] for k in ("estimate", "se", "df", "p", "n", "lambda")} | {"tree": i})
    pt = pd.DataFrame(rows)
    return pt, pooled(pt, sesoi)


def run_primary_analysis(df, mcc, posterior, sesoi):
    """Sec. 6-7 in full: MCC fit + decision, posterior pooling + 'robust' label."""
    m, res = fit_primary(df, mcc, sesoi)
    pt, pool = primary_over_trees(df, posterior, sesoi) if posterior else (None, None)
    robust = pool is not None and pool["outcome"] == m["outcome"]
    return {"mcc": m, "coef_mcc": res["coef"].to_dict("records"), "posterior": pool,
            "robust_trees": robust, "outcome": m["outcome"] + (" (robust over trees)" if robust else "")}, pt


# ============================================================== 5. robustness runner (sec. 8)

class Ctx:
    def __init__(self, cfg, td, mcc, posterior, synthetic, rebuild=None):
        self.cfg, self.td, self.mcc, self.posterior, self.synthetic, self.rebuild = cfg, td, mcc, posterior, synthetic, rebuild
        self.sesoi = cfg["sesoi"]; self._cache = {}

    def trees(self, files, n):
        k = (tuple(files), n)
        if k not in self._cache:
            per = max(1, n // len(files)) if n else None
            self._cache[k] = [t for f in files for t in pgls.read_trees(self.td / f, per)]
        return self._cache[k]


def _brief(t):
    return {k: t[k] for k in ("estimate", "se", "ci_low", "ci_high", "p", "n", "lambda", "outcome") if k in t}


def R1_fbd_trees(ctx, df):
    trees = ctx.trees(ctx.cfg["posterior"]["fbd_crown_stem_200"], ctx.cfg["robustness"]["max_trees"])
    _, s = primary_over_trees(df, trees, ctx.sesoi); return s


def R2_borowiec_tree(ctx, df):
    raise NotImplementedError("Borowiec 2025 tree (Zenodo 14338504) not yet in R2; needs tree + one-species-per-genus resampler")


def R3_other_size(ctx, df):
    out = {}
    d = df.dropna(subset=["log10_soft"]).reset_index(drop=True)
    if len(d) > 10:
        out["soft_tissue"] = _brief(fit_primary(d, ctx.mcc, ctx.sesoi, size="log10_soft")[0])
    out["head_volume"] = "not_implemented: head volume masks not produced yet"
    return out


def R4_drop_subfamily(ctx, df):
    out = {}
    for sf, g in df.groupby("subfamily"):
        if len(g) < ctx.cfg["robustness"]["min_subfamily_n"]:
            continue
        d = df[df.subfamily != sf].reset_index(drop=True)
        out[sf] = _brief(fit_primary(d, ctx.mcc, ctx.sesoi)[0])
    return out


def R5_unstained_only(ctx, df):
    return _brief(fit_primary(df[df.stain == "unstained"].reset_index(drop=True), ctx.mcc, ctx.sesoi)[0])


def _rebuilt(ctx, df, **opts):
    """R6/R13 need the species table rebuilt from specimens; on synthetic data only the builder is exercised."""
    if ctx.rebuild is None:
        raise NotImplementedError("no specimen-level rebuild available")
    alt, info = ctx.rebuild(**opts)
    if ctx.synthetic:
        return None, {"status": "builder_exercised_on_real_brain_table_no_fit", **info}
    return alt, info


def R6_all_workers(ctx, df):
    alt, info = _rebuilt(ctx, df, ok_col="brain_ok")
    return info if alt is None else _brief(fit_primary(alt, ctx.mcc, ctx.sesoi)[0])


def R7_no_qualified_names(ctx, df):
    return _brief(fit_primary(df[~df.qualified_name].reset_index(drop=True), ctx.mcc, ctx.sesoi)[0])


def R8_no_supercolonial_parasites(ctx, df):
    raise NotImplementedError("needs Bell-Roberts 2024 table 21 (supercolonial / social parasite flags) joined to species keys")


def R9_colony_variants(ctx, df):
    out = {}
    for col in ("log10_colony_br", "log10_colony_max_owners"):
        d = df.dropna(subset=[col]).reset_index(drop=True)
        out[col] = _brief(fit_primary(d, ctx.mcc, ctx.sesoi, colony=col)[0])
    return out


def R10_genus_colony(ctx, df):
    d = df.dropna(subset=["log10_colony_genus"]).reset_index(drop=True)
    r = _brief(fit_primary(d, ctx.mcc, ctx.sesoi, colony="log10_colony_genus")[0])
    r["note"] = "approximation: genus median of species-level owners' S3 means in our species table; extension to species without species-level colony not yet built"
    return r


def R11_no_genus_placed(ctx, df):
    return _brief(fit_primary(df[df.how != "genus_placed"].reset_index(drop=True), ctx.mcc, ctx.sesoi)[0])


def R12_brownian_and_owners_vcv(ctx, df):
    out = {"lambda_1": _brief(fit_primary(df, ctx.mcc, ctx.sesoi, lam=1.0)[0])}
    trees = ctx.trees(ctx.cfg["posterior"]["fbd_crown_stem_200"], ctx.cfg["robustness"]["max_trees"])
    ok = [t for t in trees if df.tip.isin(set(t.tip_index)).all()]
    if ok:
        C = sum(t.vcv(list(df.tip)) for t in ok) / len(ok) / 200.0; np.fill_diagonal(C, 1.0)
        X, names, _ = model_design(df, colony="log10_colony_any")
        t = pgls.term(pgls.gls(df.log10_brain.to_numpy(float), X, C, names), "log10_colony_any")
        out["owners_meanVCV_div200"] = {**_brief(t), "n_trees": len(ok)}
    return out


def R13_preservation(ctx, df):
    alt, info = _rebuilt(ctx, df, include_badly_preserved=True)
    if alt is None:
        info["year_covariate"] = _brief(fit_primary(df.dropna(subset=["year"]).reset_index(drop=True), ctx.mcc, ctx.sesoi, extra=("year",))[0]) \
            if ctx.synthetic else None
        return info
    d = alt.dropna(subset=["year"]).reset_index(drop=True)
    return _brief(fit_primary(d, ctx.mcc, ctx.sesoi, extra=("year",))[0])


def R14_ols(ctx, df):
    X, names, _ = model_design(df, colony="log10_colony_any")
    return _brief(pgls.term(pgls.ols(df.log10_brain.to_numpy(float), X, names), "log10_colony_any"))


ROBUSTNESS = {"R1": R1_fbd_trees, "R2": R2_borowiec_tree, "R3": R3_other_size, "R4": R4_drop_subfamily,
              "R5": R5_unstained_only, "R6": R6_all_workers, "R7": R7_no_qualified_names,
              "R8": R8_no_supercolonial_parasites, "R9": R9_colony_variants, "R10": R10_genus_colony,
              "R11": R11_no_genus_placed, "R12": R12_brownian_and_owners_vcv, "R13": R13_preservation, "R14": R14_ols}


def run_robustness(ctx, df, which):
    out = {}
    for k in which:
        t0 = time.time()
        try:
            out[k] = {"status": "ok", "result": ROBUSTNESS[k](ctx, df)}
        except NotImplementedError as e:
            out[k] = {"status": "not_implemented", "reason": str(e)}
        except Exception as e:  # a failing check must not kill the run; it is reported
            out[k] = {"status": "error", "error": repr(e)[:300]}
        out[k]["seconds"] = round(time.time() - t0, 1)
        log("robustness", k, out[k]["status"], out[k]["seconds"], "s")
    return out


# ============================================================== 4b. synthetic check

def synthetic_frame(cfg, spec, spp, body_rule_specimens, match):
    """Real species set (prereg brain set in the MCC tree), real colony sizes, stain and step class from the specimen
    table, real species-mean log10 body where the body rule gives one (else simulated), real names/subfamilies/years."""
    w = spec[(spec.caste == "worker") & spec.species_key.notna() & ~spec.caste_detail.isin(cfg["exclude_caste_detail"])].copy()
    w["vox_class"] = w.voxel_um.round(2).map(VOX_CLASSES).fillna(w.voxel_um.astype(str))
    w["qualified_name"] = w.name.astype(str).str.contains(r"\b(?:cf|aff|nr)\b\.?", case=False, regex=True)
    w["year"] = pd.to_datetime(w.date_collected, errors="coerce").dt.year
    rows = []
    for sp, g in w.groupby("species_key"):
        stain = "unstained" if (g.stained == "unstained").any() else "iodine"
        g = g[g.stained == stain]; vc = g.vox_class.value_counts()
        rows.append({"species_key": sp, "subfamily": g.subfamily.iloc[0], "genus": sp.split("_")[0], "stain": stain,
                     "vox_class": sorted(vc[vc == vc.max()].index)[0], "n_specimens": len(g),
                     "qualified_name": bool(g.qualified_name.any()), "year": g.year.mean()})
    d = pd.DataFrame(rows)
    d = d.merge(match.dropna(subset=["tip"])[["species_key", "tip", "how"]], on="species_key")
    d = attach_colony(d, spp).dropna(subset=["log10_colony_any"]).reset_index(drop=True)
    bb = body_rule_specimens.dropna(subset=["log10_body"]).groupby("species_key").log10_body.mean()
    d["log10_body_real"] = d.species_key.map(bb)
    return d


def simulate(d, V, beta2, sc, rng):
    """log10 brain = a + b1*body + beta2*colony + iodine shift + step shifts + phylo-lambda residual + measurement noise."""
    n = len(d); Vn = V / np.mean(np.diag(V))
    Lb = np.linalg.cholesky(Vn)
    body = d.log10_body_real.to_numpy(float).copy()
    miss = np.isnan(body)
    mu, sd = (np.nanmean(body), np.nanstd(body)) if (~miss).sum() > 5 else (0.0, 0.4)
    body[miss] = (mu + sd * (Lb @ rng.standard_normal(n)))[miss]
    L = np.linalg.cholesky(pgls.lambda_matrix(Vn, sc["lambda"]))
    col = d.log10_colony_any.to_numpy(float)
    y = (sc["intercept"] + sc["body_slope"] * body + beta2 * col + sc["iodine_shift"] * (d.stain == "iodine").to_numpy(float)
         + sc["vox122_shift"] * (d.vox_class == "1.22").to_numpy(float) + sc["vox611_shift"] * (d.vox_class == "6.11").to_numpy(float)
         + sc["resid_sd"] * (L @ rng.standard_normal(n)) + sc["me_sd"] * rng.standard_normal(n))
    soft = body - 0.08 + 0.03 * rng.standard_normal(n)  # synthetic soft-tissue size for R3
    return d.assign(log10_body=body, log10_brain=y, log10_soft=soft)


def synthetic_check(cfg, d, mcc, posterior, ctx_factory):
    sc = cfg["synthetic"]; rng = np.random.default_rng(sc["seed"]); sesoi = cfg["sesoi"]
    V = mcc.vcv(list(d.tip))
    rows = []
    for beta2 in sc["beta2_values"]:
        for k in range(sc["nsim"]):
            s = simulate(d, V, beta2, sc, rng)
            r, _ = fit_primary(s, mcc, sesoi, V=V)
            rows.append({"beta2_true": beta2, "rep": k, "estimate": r["estimate"], "se": r["se"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"], "lambda": r["lambda"], "size_slope": r["size_slope"], "outcome": r["outcome"]})
    reps = pd.DataFrame(rows)
    summ = []
    for b, g in reps.groupby("beta2_true"):
        summ.append({"beta2_true": b, "nsim": len(g), "mean_estimate": g.estimate.mean(), "bias": g.estimate.mean() - b,
                     "rmse": float(np.sqrt(((g.estimate - b) ** 2).mean())), "mean_se": g.se.mean(), "sd_estimate": g.estimate.std(),
                     "coverage_95": float(((g.ci_low <= b) & (b <= g.ci_high)).mean()), "mean_lambda": g["lambda"].mean(),
                     "mean_body_slope": g.size_slope.mean(), "outcomes": g.outcome.value_counts(normalize=True).round(3).to_dict()})
    # one full pass of the preregistered pipeline (MCC + posterior + Haller + robustness) on one synthetic dataset
    b_demo = sc["demo_beta2"]
    s = simulate(d, V, b_demo, sc, np.random.default_rng(sc["seed"] + 1))
    full, pt = run_primary_analysis(s, mcc, posterior, sesoi)
    h, _ = haller(s, mcc, cfg["haller"])
    rob = run_robustness(ctx_factory(), s, cfg["robustness"]["run"])
    ok = all(abs(x["bias"]) < sc["max_abs_bias"] and abs(x["coverage_95"] - 0.95) < sc["coverage_tol"] for x in summ)
    return {"n_species": int(len(d)), "n_body_real": int(d.log10_body_real.notna().sum()), "settings": sc,
            "recovery": summ, "recovery_pass": bool(ok),
            "demo": {"beta2_true": b_demo, "primary": full, "haller_on_synthetic": h,
                     "haller_body_slope_true": sc["body_slope"]}}, reps, rob, pt, s


# ============================================================== R cross-check (prereg sec. 6: phylolm / nlme corPagel)

def r_crosscheck(cfg, out, name, tree_path, df, size, colony, py_term, focus):
    """Re-fit one model with phylolm and nlme::gls(corPagel) via src/analysis/phylolm_check.R; differences vs Python."""
    from stages.cuticle_replication import run_r
    d = df.copy()
    xs = [size] + ([colony] if colony else [])
    for col, cond in (("d_iodine", d.stain == "iodine"), ("d_v122", d.vox_class == "1.22"), ("d_v611", d.vox_class == "6.11")):
        if cond.any() and not cond.all():
            d[col] = cond.astype(float); xs.append(col)
    rc = out / f"r_check_{name}"; rc.mkdir(exist_ok=True)
    d[["species_key", "tip"]].to_csv(rc / "match.csv", index=False)
    d[["species_key", "log10_brain"] + xs].to_csv(rc / "data.csv", index=False)
    try:
        if not run_r(cfg, "phylolm_check.R", [tree_path, rc / "match.csv", rc / "data.csv", "log10_brain ~ " + " + ".join(xs),
                                              rc / "phylolm.csv"], rc / "R_log.txt"):
            return {"status": "skipped_no_R"}
    except Exception as e:
        return {"status": "error", "error": repr(e)[:300]}
    r = pd.read_csv(rc / "phylolm.csv")
    eng = {e: float(abs(g.set_index("term").loc[focus, "estimate"] - py_term["estimate"])) for e, g in r.groupby("engine")}
    return {"status": "ok", "engines": sorted(eng), "pass_1e-4": bool(eng) and all(v < 1e-4 for v in eng.values()), **{e: {"d_estimate": float(g.set_index("term").loc[focus, "estimate"] - py_term["estimate"]),
                                   "d_se": float(g.set_index("term").loc[focus, "se"] - py_term["se"]),
                                   "d_lambda": float(g["lambda"].iloc[0] - py_term["lambda"])} for e, g in r.groupby("engine")}}


# ============================================================== figures

def fig_haller(df, h, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    col = {"unstained": "#2a78d6", "iodine": "#eb6834"}
    for st, g in df.groupby("stain"):
        ax.scatter(g.log10_body, g.log10_brain, s=36, c=col.get(st, "#888"), edgecolors="white", linewidths=1.2,
                   label=f"{st} (n={len(g)})", zorder=3)
    if h.get("slope") is not None:
        coef = {c["term"]: c["estimate"] for c in h["coef"]}
        xs = np.linspace(df.log10_body.min(), df.log10_body.max(), 50)
        ax.plot(xs, coef["(Intercept)"] + coef["log10_body"] * xs, color="#333", lw=2, zorder=2,
                label=f"PGLS slope {h['slope']:.2f} [{h['ci_low']:.2f}, {h['ci_high']:.2f}], λ={h['lambda']:.2f}")
    ax.set_xlabel("log10 body volume, mm³"); ax.set_ylabel("log10 brain volume, mm³")
    ax.set_title(f"Haller's rule control (gate 3): {h.get('status')}, n={h['n_species']} species", fontsize=10)
    ax.grid(color="#ddd", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_synthetic(reps, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bs = sorted(reps.beta2_true.unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.boxplot([reps[reps.beta2_true == b].estimate for b in bs], positions=range(len(bs)), widths=0.5,
               medianprops={"color": "#2a78d6", "lw": 2}, flierprops={"markersize": 3})
    for i, b in enumerate(bs):
        ax.hlines(b, i - 0.35, i + 0.35, color="#eb6834", lw=2, label="true β₂" if i == 0 else None)
    ax.set_xticks(range(len(bs)), [f"{b:+.2f}" for b in bs]); ax.set_xlabel("true β₂ (log10 colony)")
    ax.set_ylabel("estimated β₂, MCC PGLS-λ"); ax.set_title("Synthetic recovery on the real tree / species / colony sizes", fontsize=10)
    ax.grid(axis="y", color="#ddd", lw=0.6); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ============================================================== main

def main(cfg, out):
    t0 = time.time(); c = r2.client()
    inp = out.parent / "inputs"; inp.mkdir(exist_ok=True)
    src = {}
    for n in ("specimens.csv", "species.csv"):
        k = f"ant-twin/runs/{cfg['species_table_run']}/outputs/{n}"; r2_get(c, k, inp / n); src[n] = {"r2": k, "sha256": sha(inp / n)}
    p = r2_get(c, cfg["s1_r2_key"], inp / "S1.xlsx"); src["S1"] = {"r2": cfg["s1_r2_key"], "sha256": sha(p)}
    body_run = resolve_body_run(c, cfg)
    k = f"ant-twin/runs/{body_run}/outputs/body.jsonl"; r2_get(c, k, inp / "body.jsonl"); src["body"] = {"r2": k, "sha256": sha(inp / "body.jsonl")}
    log("body run:", body_run)
    spec, spp = pd.read_csv(inp / "specimens.csv"), pd.read_csv(inp / "species.csv")
    s1 = pd.read_excel(inp / "S1.xlsx")
    body = pd.DataFrame(read_jsonl(inp / "body.jsonl"))
    brains = load_brains(c, cfg, inp, src)
    td = trees_dir(cfg, c)
    for f in [cfg["mcc"]]:
        src[f] = {"path": str(td / f), "sha256": sha(td / f)}
    (out / "sources.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
    blind = cfg.get("blind", True)

    # ---- 1. specimens
    d, funnel = build_specimens(cfg, spec, brains, body, s1)
    log("funnel:", json.dumps(funnel))

    # ---- tree matching: prereg brain set (workers with species-level colony; = cuticle stage) and all worker species
    mcc = pgls.read_trees(td / cfg["mcc"])[0]
    wk = spec[spec.caste == "worker"].dropna(subset=["species_key"])
    brain_set = sorted(wk.dropna(subset=["colony_any"]).species_key.unique())
    m_primary = pgls.match_species(brain_set, mcc.tip_labels)
    m_all = pgls.match_species(sorted(wk.species_key.unique()), mcc.tip_labels)
    m_primary.to_csv(out / "tree_match_primary_set.csv", index=False)

    # ---- 2. species
    def build_species(ok_col="main_ok", include_badly_preserved=False, match=m_primary):
        dd = d
        if include_badly_preserved:
            pats = [p for p in cfg["exclude_problems"] if "badly" not in p]
            items = dd.problems.map(problem_items)
            dmg = items.map(lambda it: matches_any(it, pats))
            base = dd.qc_review_pass & ~dd.qc_autoflag_fail & ~dd.flag_empty & dd.species_key.notna() & ~dmg
            dd = dd.assign(_ok=base & (~dd.excl_caste if ok_col == "main_ok" else True)); ok_col = "_ok"
        sp = species_means(dd, ok_col, cfg["pair_brain_body"], cfg["imputed_rel_error"])
        if sp.empty:
            return sp
        sp = sp.merge(match[["species_key", "tip", "how"]], on="species_key", how="left")
        return attach_colony(sp, spp)

    sp_all = build_species(match=m_all)  # Haller set: no colony requirement
    sp_prim = build_species()            # primary set: prereg brain set match
    hal_df = sp_all.dropna(subset=["tip", "log10_brain", "log10_body"]).reset_index(drop=True) if len(sp_all) else sp_all
    prim_df = sp_prim.dropna(subset=["tip", "log10_brain", "log10_body", "log10_colony_any"]).reset_index(drop=True) if len(sp_prim) else sp_prim
    counts = {"species_with_qc_brain": int(d[d.main_ok].species_key.nunique()),
              "species_with_qc_brain_and_body": int(len(sp_all)),
              "species_brain_body_in_tree": int(len(hal_df)),
              "species_brain_body_colony_in_tree": int(len(prim_df)),
              "species_brain_body_colony_in_tree_unstained": int((prim_df.stain == "unstained").sum()) if len(prim_df) else 0,
              "prereg_brain_set_species": len(brain_set), "prereg_brain_set_in_tree": int(m_primary.tip.notna().sum())}
    log("counts:", json.dumps(counts))

    # outputs (blind: no colony values next to real brains)
    colony_cols = [x for x in d.columns if "colony" in x]
    spec_out = d.assign(has_colony=d[colony_cols[0]].notna() if colony_cols else False)
    sp_out = sp_all.assign(has_colony=sp_all.log10_colony_any.notna()) if len(sp_all) else sp_all
    if blind:
        spec_out = spec_out.drop(columns=colony_cols + ["n_worker_castes_br", "n_queens_br"], errors="ignore")
        sp_out = sp_out.drop(columns=[x for x in sp_out.columns if "colony" in x], errors="ignore")
    spec_out.to_csv(out / "specimens_table.csv", index=False)
    sp_out.to_csv(out / "species_table.csv", index=False)
    brain_hash = hashlib.sha256(sp_out.drop(columns=["has_colony"], errors="ignore").to_csv(index=False).encode()).hexdigest()

    # ---- 3. Haller (gate 3)
    h, _ = haller(hal_df, mcc, cfg["haller"]) if len(hal_df) else ({"status": "no_data", "n_species": 0}, None)
    h["species_set"] = "all worker species with QC brain + body in MCC tree (no colony requirement)"
    if h.get("slope") is not None:
        h["r_crosscheck"] = r_crosscheck(cfg, out, "haller", td / cfg["mcc"], hal_df, "log10_body", None,
                                         {"estimate": h["slope"], "se": h["se"], "lambda": h["lambda"]}, "log10_body")
        log("R cross-check (Haller):", json.dumps(h["r_crosscheck"]))
    (out / "haller.json").write_text(json.dumps(h, indent=2, default=float), encoding="utf-8")
    log("haller:", json.dumps({k: h.get(k) for k in ("status", "n_species", "slope", "ci_low", "ci_high", "r2_ols_same_terms", "lambda")}, default=float))
    if h.get("slope") is not None:
        fig_haller(hal_df, h, out / "fig_haller.png")

    # ---- posterior trees for the primary model
    sc = cfg["synthetic"]
    post_files = cfg["posterior"][cfg["primary_posterior_set"]]
    n_post = None if cfg.get("run_primary") else sc["posterior_trees"]
    posterior = [t for f in post_files for t in pgls.read_trees(td / f, n_post)]
    log("posterior trees loaded:", len(posterior))

    def rebuild(ok_col="main_ok", include_badly_preserved=False):
        alt = build_species(ok_col=ok_col, include_badly_preserved=include_badly_preserved)
        alt = alt.dropna(subset=["tip", "log10_brain", "log10_body", "log10_colony_any"]).reset_index(drop=True) if len(alt) else alt
        return alt, {"n_species": int(len(alt))}

    # ---- 4b. synthetic check (always)
    syn_d = synthetic_frame(cfg, spec, spp, build_body_rule_specimens(cfg, spec, body, s1), m_primary)
    syn, reps, rob_syn, syn_pt, syn_demo = synthetic_check(cfg, syn_d, mcc, posterior,
                                                 lambda: Ctx(cfg, td, mcc, posterior, synthetic=True, rebuild=rebuild))
    syn["r_crosscheck_demo_primary"] = r_crosscheck(cfg, out, "synthetic_primary", td / cfg["mcc"], syn_demo, "log10_body",
                                                    "log10_colony_any", syn["demo"]["primary"]["mcc"], "log10_colony_any")
    log("R cross-check (synthetic primary):", json.dumps(syn["r_crosscheck_demo_primary"]))
    reps.to_csv(out / "synthetic_reps.csv", index=False)
    if syn_pt is not None:
        syn_pt.to_csv(out / "synthetic_demo_posterior.csv", index=False)
    (out / "synthetic_check.json").write_text(json.dumps(syn, indent=2, default=float), encoding="utf-8")
    (out / "robustness_synthetic.json").write_text(json.dumps(rob_syn, indent=2, default=float), encoding="utf-8")
    fig_synthetic(reps, out / "fig_synthetic_recovery.png")
    log("synthetic recovery:", json.dumps(syn["recovery"], default=float))

    # ---- 4a. primary on real brains (guarded)
    primary = {"executed": False, "reason": "run_primary is false (dry run; preregistered test not looked at)"}
    if cfg.get("run_primary"):
        if blind:
            raise RuntimeError("run_primary requires blind=false (and OSF registration done)")
        if h.get("status") != "pass" and not cfg.get("override_gate3"):
            primary = {"executed": False, "reason": f"gate 3 status {h.get('status')}: primary not tested (prereg sec. 5)"}
        else:
            res, pt = run_primary_analysis(prim_df, mcc, posterior, cfg["sesoi"])
            pt.to_csv(out / "primary_posterior.csv", index=False)
            primary = {"executed": True, **res}
            primary["robustness"] = run_robustness(Ctx(cfg, td, mcc, posterior, synthetic=False, rebuild=rebuild), prim_df, cfg["robustness"]["run"])
            prim_df.to_csv(out / "primary_data.csv", index=False)
    (out / "primary.json").write_text(json.dumps(primary, indent=2, default=float), encoding="utf-8")

    summary = {"funnel": funnel, "counts": counts, "brain_table_sha256": brain_hash, "body_run": body_run,
               "brain_sources": cfg["brain_sources"], "qc": cfg["qc"], "blind": blind,
               "haller": {k: h.get(k) for k in ("status", "n_species", "slope", "se", "ci_low", "ci_high", "p", "lambda",
                                                "r2_ols_same_terms", "r2_ols_body_only", "checks", "dropped_constant_terms", "resid_sd_pgls")},
               "synthetic": {"recovery_pass": syn["recovery_pass"], "recovery": syn["recovery"],
                             "demo_outcome": syn["demo"]["primary"]["outcome"],
                             "demo_haller_slope": syn["demo"]["haller_on_synthetic"].get("slope"),
                             "r_crosscheck_demo_primary": syn["r_crosscheck_demo_primary"]},
               "haller_r_crosscheck": h.get("r_crosscheck"),
               "robustness_synthetic_status": {k: v["status"] for k, v in rob_syn.items()},
               "primary": {"executed": primary["executed"]}, "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    log("summary:", json.dumps(summary, default=float))


def build_body_rule_specimens(cfg, spec, body, s1):
    """Body-rule log10 body (mm^3) for every eligible worker, independent of brains (used for synthetic bodies)."""
    w = spec[(spec.caste == "worker") & spec.species_key.notna()].copy()
    w["key"] = [spec_key(f, v) for f, v in zip(w.folder, w.vial)]
    s1w = s1[s1.caste == "worker"].drop_duplicates("specimen_code").set_index("specimen_code")
    bv = body.drop_duplicates("key").set_index("key")
    b_s1 = w.specimen_code.map(s1w.body_volume)
    ok = (w.key.map(bv.status) == "ok") & (w.key.map(bv.guarded_slices) < cfg["body_guard_max"])
    b = b_s1.where(b_s1.notna(), w.key.map(bv.body_um3).where(ok))
    dmg = w.problems.map(problem_items).map(lambda it: matches_any(it, cfg["body_only_problems"] + cfg["exclude_problems"]))
    w["log10_body"] = np.log10(b.where(~dmg) / UM3_PER_MM3)
    return w[~w.caste_detail.isin(cfg["exclude_caste_detail"])]
