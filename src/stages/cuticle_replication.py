"""Stage cuticle_replication: validate our tree matching + PGLS code on the owners' published result before brain data exist.

Target: Matte et al. 2025 (Sci Adv 11: eadx8068), Table S2, model 2 - larger colonies, lower worker cuticle volume
relative to body volume (colony term -0.040914 per SD of log10 colony size, SE 0.007961, n = 325, p = 4.84e-7).

A. Owners' own R code (src/analysis/owners_cuticle_replication.R; needs Rscript): their data assembly, their 200 FBD trees,
   their nlme::gls with a fixed mean-VCV correlation. Run as published and with their tip-assignment bug fixed.
   Our Python gls() is then re-fitted on the exact R design/covariance -> must agree to numerical precision.
B. Our machinery: owners' S1 volumes joined to OUR specimen table (species keys, colony size), OUR name matching to the
   Economo 2018 15K tree, Pagel-lambda PGLS (pgls.py) on the MCC tree and across posterior trees; phylolm cross-check.
Inputs are fetched inside the run (Zenodo, R2) and hashed; Economo trees from cfg tree_dir or the R2 copy.
"""
import io, os, json, time, shutil, hashlib, zipfile, pathlib, subprocess
import numpy as np, pandas as pd
import r2
from analysis import pgls
from stages.specimen_table import key as name_key, fetch, SRC

PUBLISHED = {"source": "Matte et al. 2025 Sci Adv, Table S2 model 2 (Economo 2018 FBD 200 trees)",
             "colony_per_sd": -0.040914, "se": 0.007961, "t": -5.138942, "p": 4.84e-07, "n": 325, "df": 317,
             "log10_body": 0.97366,
             "table_S3_borowiec_tree": {"colony_per_sd": -0.051183, "se": 0.01119, "p": 1.17e-05, "n": 128}}
ANALYSIS = pathlib.Path(__file__).resolve().parent.parent / "analysis"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def trees_dir(cfg, c):
    d = pathlib.Path(cfg["tree_dir"])
    if d.exists():
        return d
    d = pathlib.Path.home() / ".cache" / "ant-twin" / "economo2018"; d.mkdir(parents=True, exist_ok=True)
    for f in [cfg["mcc"]] + [x for v in cfg["posterior"].values() for x in v]:
        if not (d / f).exists():
            c.download_file(r2.BUCKET, f"{cfg['tree_r2_prefix']}/{f}", str(d / f)); print("tree from R2:", f)
    return d


def run_r(cfg, script, args, log):
    rs = cfg.get("rscript")
    if not rs or not pathlib.Path(rs).exists():
        print("[R] Rscript not available -> R steps skipped"); return False
    env = dict(os.environ); env["R_LIBS_USER"] = cfg.get("r_libs", "")
    p = subprocess.run([rs, str(ANALYSIS / script)] + [str(a) for a in args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    lines = [l for l in (p.stdout + p.stderr).splitlines() if "New names" not in l and "`` ->" not in l]
    log.write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines[-40:]))
    if p.returncode:
        raise RuntimeError(f"{script} failed ({p.returncode})")
    return True


def check_owners_gls(rdir):
    """Re-fit owners' model 2 with pgls.gls() on the exact data + correlation matrix exported from R."""
    tab = pd.read_csv(rdir / "owners_replication_table.csv")
    out = {}
    for variant in ("owners_code", "fixed_tip_assignment"):
        d = pd.read_csv(rdir / f"model2_data_{variant}.csv")
        cov = pd.read_csv(rdir / f"model2_cov_{variant}.csv", index_col=0)
        assert list(cov.index) == list(d["genus.species"]), "row order differs"
        C = cov.to_numpy(float).copy(); np.fill_diagonal(C, 1.0)  # corSymm uses the lower triangle, unit diagonal
        d = d.assign(log10_body=np.log10(d.body_volume))
        X, names = pgls.design(d, ["mean_ann_temp", "colony_size", "spine_number"],
                               {"diet": ["omnivore", "predator", "herbivore", "fungivore"]})
        X = np.column_stack([X, d.log10_body]); names.append("log10(body_volume)")
        res = pgls.gls(np.log10(d.cuticle_volume), X, C, names)
        rr = tab[(tab.variant == variant) & (tab.model == "model2")].set_index("term")
        py = res["coef"].set_index("term")
        rn = {"dietpredator": "dietpredator", "dietherbivore": "dietherbivore", "dietfungivore": "dietfungivore"}
        diffs = {t: {"d_est": float(py.loc[t, "estimate"] - rr.loc[rn.get(t, t), "Value"]),
                     "d_se": float(py.loc[t, "se"] - rr.loc[rn.get(t, t), "Std.Error"])} for t in names}
        out[variant] = {"n": res["n"], "logLik_py": res["logLik"], "logLik_R": float(rr.logLik.iloc[0]),
                        "max_abs_diff_estimate": max(abs(v["d_est"]) for v in diffs.values()),
                        "max_abs_diff_se": max(abs(v["d_se"]) for v in diffs.values()),
                        "colony_per_sd": float(py.loc["colony_size", "estimate"]), "colony_se": float(py.loc["colony_size", "se"]),
                        "colony_p": float(py.loc["colony_size", "p"]),
                        "sd_log10_colony": float(rr.sd_log10_colony.iloc[0])}
    return out


def build_species(inp, spec, spp):
    """Owners' S1 worker volumes -> our species keys; species means of log10 volumes; colony sizes; owners' covariates."""
    od = inp / "owners"
    s1 = pd.read_excel(next(od.rglob("Dataset_S1*.xlsx")))
    w = s1[s1.caste == "worker"].merge(spec[["specimen_code", "species_key", "stained", "folder"]].drop_duplicates("specimen_code"),
                                       on="specimen_code", how="left")
    stats_join = {"s1_workers": int(len(w)), "s1_workers_with_our_species_key": int(w.species_key.notna().sum())}
    w = w.dropna(subset=["species_key"]).copy()
    w["minor"] = ~w.subcaste.isin(["major", "media", "intercaste"])
    w["lb"], w["lc"] = np.log10(w.body_volume), np.log10(w.cuticle_volume)
    w["ls"] = np.log10(w.body_volume - w.cuticle_volume)  # soft tissue = body minus cuticle

    def means(x):
        return x.groupby("species_key").agg(log10_body=("lb", "mean"), log10_cuticle=("lc", "mean"), log10_soft=("ls", "mean"),
                                            n_specimens=("lb", "size"))
    d = means(w).join(means(w[w.minor]).add_suffix("_minor"), how="left")
    d = d.join(spp.set_index("species_key")[["colony_any", "colony_mean_owners", "subfamily"]], how="left")
    d["genus"] = d.index.str.split("_").str[0]
    # owners' genus-level colony size: species median of 'Mean Size', then genus median (their load_trait_data.R)
    s3 = pd.read_excel(next(od.rglob("Dataset_S3*.xlsx")))
    g = (s3.groupby("Valid Species Name").agg(genus=("Valid Genus", "first"), cs=("Mean Size", "median"))
         .groupby("genus").cs.median())
    d["colony_genus_owners"] = d.genus.map({k.lower(): v for k, v in g.items()})
    s5 = pd.read_excel(next(od.rglob("Dataset_S5*.xlsx"))); s5["k"] = s5["genus.species"].map(name_key)
    s6 = pd.read_excel(next(od.rglob("Dataset_S6*.xlsx"))); s6["k"] = s6["genus.species"].map(name_key)
    s4 = pd.read_excel(next(od.rglob("Dataset_S4*.xlsx")))
    d["mean_ann_temp"] = d.index.map(s5.dropna(subset=["k"]).drop_duplicates("k").set_index("k").mean_ann_temp)
    d["diet"] = d.index.map(s6.dropna(subset=["k"]).drop_duplicates("k").set_index("k").diet)
    d["spine_number"] = d.genus.map({str(k).lower(): v for k, v in s4.set_index("genus").spine_number.items()})
    for c in ("colony_any", "colony_mean_owners", "colony_genus_owners"):
        d["log10_" + c] = np.log10(d[c])
    stats_join["species_with_volumes"] = int(len(d))
    return d.reset_index(), stats_join


def match_summary(m):
    return {"species": int(len(m)), "counts": m.how.value_counts().to_dict(),
            "matched": int((m.how != "unmatched").sum()), "match_rate": float((m.how != "unmatched").mean()),
            "match_rate_by_name": float(m.how.isin(["exact", "subspecies_tip", "genus_transfer", "epithet_gender"]).mean()),
            "unmatched": m[m.how == "unmatched"].species_key.tolist(),
            "genus_placed": m[m.how == "genus_placed"].species_key.tolist(),
            "renamed": m[m.how.isin(["genus_transfer", "epithet_gender", "subspecies_tip"])][["species_key", "tip", "how"]].to_dict("records")}


def power_sim(V, x, sesoi=0.02, lam=0.8, me=0.03, nsim=1000, seed=7):
    """Power of the primary brain test on the real species set, tree and colony sizes (GLS with the true lambda).
    log10 brain = 0.7 log10 body + beta * log10 colony + s * phylo(lambda) + me * noise; body = 0.1 colony + BM."""
    from scipy.linalg import solve_triangular
    from scipy import stats as st
    rng = np.random.default_rng(seed); V = V / np.mean(np.diag(V)); n = len(x); xc = x - x.mean()
    L = np.linalg.cholesky(pgls.lambda_matrix(V, lam)); Lb = np.linalg.cholesky(V)
    q95, q975 = st.t.ppf(0.95, n - 3), st.t.ppf(0.975, n - 3)
    rows = []
    for s in (0.06, 0.10, 0.15, 0.25):
        for beta in (0.0, -0.02, -0.03, -0.04):
            est, se = np.empty(nsim), np.empty(nsim)
            for k in range(nsim):
                body = 0.1 * xc + 0.4 * (Lb @ rng.standard_normal(n))
                y = 0.7 * body + beta * xc + s * (L @ rng.standard_normal(n)) + me * rng.standard_normal(n)
                X = np.column_stack([np.ones(n), body, x])
                Xs, ys = solve_triangular(L, X, lower=True), solve_triangular(L, y, lower=True)
                b, *_ = np.linalg.lstsq(Xs, ys, rcond=None); r = ys - Xs @ b
                est[k] = b[2]; se[k] = np.sqrt(r @ r / (n - 3) * np.linalg.inv(Xs.T @ Xs)[2, 2])
            rows.append({"resid_sd": s, "beta_true": beta, "n_species": n, "mean_se": float(se.mean()),
                         "power_two_sided_0.05": float((np.abs(est / se) > q975).mean()),
                         "power_negative_ci_excludes_0": float((est + q975 * se < 0).mean()),
                         "power_equivalence_tost_sesoi": float(((est - q95 * se) > -sesoi).__and__((est + q95 * se) < sesoi).mean())})
    return pd.DataFrame(rows)


def main(cfg, out):
    t0 = time.time(); c = r2.client()
    inp = out.parent / "inputs"; inp.mkdir(exist_ok=True)
    sources = {}
    raw = fetch(SRC["owners_datasets_zip"])
    sources["owners_datasets_zip"] = {"url": SRC["owners_datasets_zip"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    zipfile.ZipFile(io.BytesIO(raw)).extractall(inp / "owners")
    for n in ("specimens.csv", "species.csv"):
        k = f"ant-twin/runs/{cfg['species_table_run']}/outputs/{n}"; c.download_file(r2.BUCKET, k, str(inp / n))
        sources[n] = {"r2": k, "sha256": sha(inp / n)}
    td = trees_dir(cfg, c)
    for f in [cfg["mcc"]] + [x for v in cfg["posterior"].values() for x in v]:
        sources[f] = {"path": str(td / f), "sha256": sha(td / f)}
    (out / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    results = {"published": PUBLISHED, "config": cfg}

    # ---- A. owners' R code + exact re-fit in Python
    rdir = out / "r_owners"; rdir.mkdir(exist_ok=True)
    if run_r(cfg, "owners_cuticle_replication.R", [inp / "owners", td, rdir], rdir / "R_log.txt"):
        tab = pd.read_csv(rdir / "owners_replication_table.csv")
        results["A_owners_R"] = {"diagnostics": json.loads((rdir / "owners_diagnostics.json").read_text()),
                                 "colony_terms": tab[tab.term == "colony_size"].to_dict("records")}
        results["A_python_vs_R_gls"] = check_owners_gls(rdir)
        print("[A] python gls vs R:", json.dumps(results["A_python_vs_R_gls"], indent=1))

    # ---- B. our machinery
    spec, spp = pd.read_csv(inp / "specimens.csv"), pd.read_csv(inp / "species.csv")
    d, join_stats = build_species(inp, spec, spp)
    results["B_join"] = join_stats
    mcc = pgls.read_trees(td / cfg["mcc"])[0]
    prim = d.dropna(subset=["log10_body", "log10_cuticle", "log10_colony_any"])
    m = pgls.match_species(prim.species_key, mcc.tip_labels)
    m.to_csv(out / "tree_match.csv", index=False)
    m[m.how == "unmatched"].species_key.to_csv(out / "tree_unmatched.txt", index=False, header=False)
    results["B_tree_match"] = match_summary(m)
    print("[B] tree match (cuticle set):", json.dumps(results["B_tree_match"]["counts"]))
    # the set the brain test will use: every AntScan worker species with a species-level colony size (specimen table v2)
    brain_sp = sorted(spec[spec.caste == "worker"].dropna(subset=["species_key", "colony_any"]).species_key.unique())
    mb = pgls.match_species(brain_sp, mcc.tip_labels); mb.to_csv(out / "tree_match_brain_set.csv", index=False)
    results["B_tree_match_brain_set"] = match_summary(mb)
    print("[B] tree match (brain set):", json.dumps({k: results["B_tree_match_brain_set"][k] for k in ("species", "counts", "match_rate", "unmatched")}))
    tipmap = m.dropna(subset=["tip"]).set_index("species_key").tip
    d = d[d.species_key.isin(tipmap.index)].copy(); d["tip"] = d.species_key.map(tipmap)
    d["how"] = d.species_key.map(m.set_index("species_key").how)
    d.to_csv(out / "species_data.csv", index=False)

    def z(x): return (x - x.mean()) / x.std()
    diet_levels = {"diet": ["omnivore", "predator", "herbivore", "fungivore"]}
    models = {  # name: (rows filter, y, numeric x, categorical, colony column); the size covariate is the first log10_* in x
        "M1_primary": (lambda x: x, "log10_cuticle", ["log10_body", "log10_colony_any"], None, "log10_colony_any"),
        # denominator check for the brain test: body volume at fixed soft tissue vs colony (cuticle share shrinks in big colonies)
        "M7_body_given_soft": (lambda x: x, "log10_body", ["log10_soft", "log10_colony_any"], None, "log10_colony_any"),
        "M2_minor_workers": (lambda x: x, "log10_cuticle_minor", ["log10_body_minor", "log10_colony_any"], None, "log10_colony_any"),
        "M3_no_genus_placed": (lambda x: x[x.how != "genus_placed"], "log10_cuticle", ["log10_body", "log10_colony_any"], None, "log10_colony_any"),
        "M4_owners_S3_only": (lambda x: x, "log10_cuticle", ["log10_body", "log10_colony_mean_owners"], None, "log10_colony_mean_owners"),
        "M5_genus_level_colony": (lambda x: x, "log10_cuticle", ["log10_body", "log10_colony_genus_owners"], None, "log10_colony_genus_owners"),
        "M6_owners_covariates": (lambda x: x.assign(temp_z=z(x.mean_ann_temp), spine_z=z(x.spine_number)),
                                 "log10_cuticle", ["temp_z", "spine_z", "log10_body", "log10_colony_any"], diet_levels, "log10_colony_any"),
    }

    def prep(name):
        f, y, xs, cat, col = models[name]
        need = [y] + [x for x in xs if not x.endswith("_z")] + list((cat or {}).keys()) + (["mean_ann_temp", "spine_number"] if cat else [])
        x = f(d.dropna(subset=need).reset_index(drop=True))
        X, names = pgls.design(x, xs, cat)
        return x, x[y].to_numpy(float), X, names, col

    rows = []
    for name in models:
        x, y, X, names, col = prep(name)
        V = mcc.vcv(list(x.tip))
        sd = float(x[col].std())
        for eng, res in (("pgls_lambda", pgls.pgls_lambda(y, X, V, names)), ("pgls_bm", pgls.pgls_lambda(y, X, V, names, lam=1.0)),
                         ("ols", pgls.ols(y, X, names))):
            r = pgls.term(res, col); r.update(model=name, engine=eng, tree="MCC " + cfg["mcc"], sd_log10_colony=sd,
                                              per_sd=r["estimate"] * sd, per_sd_ci_low=r["ci_low"] * sd, per_sd_ci_high=r["ci_high"] * sd,
                                              size_term=[n for n in names if n.startswith("log10_")][0],
                                              size_coef=float(res["coef"].set_index("term").loc[[n for n in names if n.startswith("log10_")][0], "estimate"]))
            rows.append(r)
        if name == "M1_primary":
            x.to_csv(out / "M1_data.csv", index=False)
    mcc_fits = pd.DataFrame(rows); mcc_fits.to_csv(out / "fits_mcc.csv", index=False)
    print(mcc_fits[["model", "engine", "n", "estimate", "se", "ci_low", "ci_high", "p", "per_sd", "lambda"]].to_string())

    # ---- phylolm / nlme corPagel cross-check of M1 on the MCC tree
    rc = out / "r_check"; rc.mkdir(exist_ok=True)
    m.to_csv(rc / "match.csv", index=False)
    if run_r(cfg, "phylolm_check.R", [td / cfg["mcc"], rc / "match.csv", out / "M1_data.csv",
                                      "log10_cuticle ~ log10_body + log10_colony_any", rc / "phylolm_M1.csv"], rc / "R_log.txt"):
        r = pd.read_csv(rc / "phylolm_M1.csv")
        py = mcc_fits[(mcc_fits.model == "M1_primary") & (mcc_fits.engine == "pgls_lambda")].iloc[0]
        results["B_python_vs_R_lambda"] = {e: {"d_estimate": float(g.set_index("term").loc["log10_colony_any", "estimate"] - py.estimate),
                                               "d_se": float(g.set_index("term").loc["log10_colony_any", "se"] - py.se),
                                               "d_lambda": float(g["lambda"].iloc[0] - py["lambda"]),
                                               "d_logLik": float(g.logLik.iloc[0] - py.logLik)} for e, g in r.groupby("engine")}
        print("[B] python vs R lambda PGLS:", json.dumps(results["B_python_vs_R_lambda"]))

    # ---- posterior trees: M1 (and owners-style fixed mean-VCV GLS on the same trees)
    x, y, X, names, col = prep("M1_primary")
    post_rows, summ = [], {}
    for setname, files in cfg["posterior"].items():
        trees = [t for f in files for t in pgls.read_trees(td / f, cfg.get("max_trees"))]
        Vsum, n_ok = None, 0
        for i, t in enumerate(trees):
            keep = x.tip.isin(set(t.tip_index)).to_numpy()
            V = t.vcv(list(x.tip[keep]))
            res = pgls.pgls_lambda(y[keep], X[keep], V, names)
            r = pgls.term(res, col); r.update(set=setname, tree=i, height=t.height); post_rows.append(r)
            if keep.all():
                Vsum = V if Vsum is None else Vsum + V; n_ok += 1
        pt = pd.DataFrame([r for r in post_rows if r["set"] == setname])
        summ[setname] = pgls.summarize_over_trees(pt)
        if n_ok:  # owners' approach on our data: mean VCV / 200 as a fixed correlation (unit diagonal)
            C = Vsum / n_ok / 200.0; np.fill_diagonal(C, 1.0)
            summ[setname]["owners_style_fixed_meanVCV_div200"] = pgls.term(pgls.gls(y, X, C, names), col)
            summ[setname]["owners_style_implied_lambda"] = float(np.median([r["height"] for r in post_rows if r["set"] == setname]) / 200.0)
        print(f"[B] {setname}:", json.dumps(summ[setname]))
    pd.DataFrame(post_rows).to_csv(out / "fits_posterior.csv", index=False)
    results["B_posterior"] = summ
    results["B_mcc"] = mcc_fits.to_dict("records")

    # ---- power of the preregistered brain test on the real brain species set, MCC tree and colony sizes
    bs = mb.dropna(subset=["tip"]).merge(spp[["species_key", "colony_any"]], on="species_key")
    xb = np.log10(bs.colony_any.to_numpy(float))
    pw = power_sim(mcc.vcv(list(bs.tip)), xb, sesoi=cfg.get("sesoi", 0.02), nsim=cfg.get("power_nsim", 1000))
    pw.to_csv(out / "power_brain_test.csv", index=False)
    results["power_brain_test"] = {"n_species": int(len(bs)), "sd_log10_colony": float(xb.std(ddof=1)),
                                   "range_log10_colony": [float(xb.min()), float(xb.max())], "table": pw.to_dict("records")}
    print("[power]\n" + pw.to_string())

    # ---- verdict
    a = results.get("A_python_vs_R_gls", {})
    m1 = mcc_fits[(mcc_fits.model == "M1_primary") & (mcc_fits.engine == "pgls_lambda")].iloc[0]
    m6 = mcc_fits[(mcc_fits.model == "M6_owners_covariates") & (mcc_fits.engine == "pgls_lambda")].iloc[0]
    results["verdict"] = {
        "python_gls_equals_R": bool(a) and all(v["max_abs_diff_estimate"] < 1e-6 and v["max_abs_diff_se"] < 1e-6 for v in a.values()),
        "python_lambda_equals_phylolm": ("B_python_vs_R_lambda" in results and abs(results["B_python_vs_R_lambda"]["phylolm"]["d_estimate"]) < 1e-4
                                         and abs(results["B_python_vs_R_lambda"]["phylolm"]["d_lambda"]) < 1e-3),
        "ours_sign_matches": bool(m1.estimate < 0 and m6.estimate < 0),
        "ours_M1_p_lt_0.05": bool(m1.p < 0.05),
        "published_per_sd_inside_ours_M6_ci": bool(m6.per_sd_ci_low <= PUBLISHED["colony_per_sd"] <= m6.per_sd_ci_high),
        "posterior_all_negative": all(s["estimate_max"] < 0 for s in summ.values()),
        "seconds": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("[verdict]", json.dumps(results["verdict"], indent=1))
