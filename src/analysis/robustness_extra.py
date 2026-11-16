"""Preregistered robustness checks that need external inputs (prereg_draft.md sec. 8):

R8  - drop supercolonial species and social parasites (Bell-Roberts et al. 2024, Supplementary Table 21).
R2  - primary model on the Borowiec et al. 2025 dated UCE phylogeny (Zenodo 14338504, genus-level sampling):
      one species per genus, picked at random, 100+ resamples; median and spread of beta2 (owners' rule).

Both reuse the primary model unchanged: stages.brain_analysis.fit_primary (PGLS-lambda, SESOI 0.02, sec. 7 rules).
"""
import re
import numpy as np, pandas as pd
from analysis import pgls
from stages.specimen_table import key as species_key_of


# ------------------------------------------------------------------ R8

def bell_roberts_flags(xlsx_path):
    """Supplementary Table 21 of Bell-Roberts et al. 2024 -> species_key, supercolonial, social_parasite (0/1)."""
    t = pd.read_excel(xlsx_path, "Supplementary Table 21", header=2)
    t = t.rename(columns={"social.parasite": "social_parasite"})
    t["species_key"] = t.species.map(species_key_of)
    t = t.dropna(subset=["species_key"]).drop_duplicates("species_key")
    for c in ("supercolonial", "social_parasite"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0).astype(int)
    return t[["species_key", "subfamily", "supercolonial", "social_parasite"]]


def r8_subset(df, flags):
    """Primary species table without species flagged supercolonial or social parasite. Returns (subset, dropped)."""
    bad = set(flags.loc[(flags.supercolonial == 1) | (flags.social_parasite == 1), "species_key"])
    m = df.species_key.isin(bad)
    dropped = df.loc[m, ["species_key", "subfamily", "log10_colony_any"]].merge(flags, on=["species_key"], how="left",
                                                                                suffixes=("", "_br"))
    return df[~m].reset_index(drop=True), dropped


# ------------------------------------------------------------------ R2: Borowiec 2025 tree

def read_nexus_tree(path):
    """First tree of a NEXUS file (FigTree/MCMCTree output); honours a TRANSLATE block; drops [&...] annotations."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    trans = {}
    m = re.search(r"translate\s+(.*?);", txt, re.I | re.S)
    if m:
        for item in m.group(1).split(","):
            p = item.strip().split(None, 1)
            if len(p) == 2:
                trans[p[0]] = p[1].strip().strip("'\"")
    m = re.search(r"\bu?tree\s+[^=]+=\s*(?:\[[^\]]*\]\s*)*(\(.*?;)", txt, re.I | re.S)
    nwk = re.sub(r"\[[^\]]*\]", "", m.group(1)) if m else txt.strip()
    t = pgls.parse_newick(nwk)
    if trans:
        t = pgls.Tree(t.parent, t.length, [trans.get(l, l) if l is not None else None for l in t.label])
    return t


def tip_genus(label):
    return re.split(r"[_ .]", str(label).strip("'\""))[0].lower()


def genus_tips(tree):
    """tree genus (lower case) -> list of tips."""
    by = {}
    for l in tree.tip_labels:
        by.setdefault(tip_genus(l), []).append(l)
    return by


def tree_genus(genus, by, transfers=None):
    """Genus as used in the tree: same name, else the first former genus from `transfers` present in the tree."""
    if genus in by:
        return genus
    return next((o for o in (transfers or {}).get(genus, []) if o in by), None)


def borowiec_resample(df, tree, fit, n_rep=100, seed=2025, transfers=None):
    """One species per (tree) genus at random, attached to one tip of that genus (random if the tree has several).
    Species of a genus split after the tree's taxonomy (`transfers`) join the group of the former genus.
    fit(d, V) -> dict with estimate/se/df/ci_low/ci_high/p/n/lambda/outcome. Returns (per-replicate DataFrame, coverage)."""
    gt = genus_tips(tree)
    d0 = df.assign(genus_l=df.species_key.str.split("_").str[0])
    d0["genus_tree"] = [tree_genus(g, gt, transfers) for g in d0.genus_l]
    have = sorted(d0.genus_tree.dropna().unique())
    cover = {"genera_in_data": int(d0.genus_l.nunique()), "tree_genera": len(gt), "tree_genera_matched": len(have),
             "species_in_matched_genera": int(d0.genus_tree.notna().sum()), "species_total": int(len(d0)),
             "genera_via_transfer": sorted(set(d0.loc[d0.genus_tree.notna() & (d0.genus_tree != d0.genus_l), "genus_l"])),
             "genera_missing_from_tree": sorted(set(d0.loc[d0.genus_tree.isna(), "genus_l"])),
             "species_per_genus_max": int(d0.genus_tree.value_counts().max())}
    rng = np.random.default_rng(seed)
    groups = {g: d0.index[d0.genus_tree == g].to_numpy() for g in have}
    rows = []
    for r in range(n_rep):
        idx = [rng.choice(groups[g]) for g in have]
        d = d0.loc[idx].reset_index(drop=True)
        d["tip"] = [gt[g][rng.integers(len(gt[g]))] for g in d.genus_tree]
        res = fit(d, tree.vcv(list(d.tip)))
        rows.append({"rep": r, **{k: res[k] for k in ("estimate", "se", "df", "ci_low", "ci_high", "p", "n", "lambda", "outcome")}})
    return pd.DataFrame(rows), cover


def summarize_resamples(rep, sesoi):
    """Median and spread of beta2 over resamples plus the pooled t-mixture interval (as for posterior trees)."""
    from stages.brain_analysis import pooled
    s = pooled(rep[["estimate", "se", "df", "p", "n", "lambda"]], sesoi)
    s.update(estimate_q025=float(rep.estimate.quantile(0.025)), estimate_q975=float(rep.estimate.quantile(0.975)),
             share_ci_excludes_0_pos=float((rep.ci_low > 0).mean()), share_ci_excludes_0_neg=float((rep.ci_high < 0).mean()),
             share_estimate_pos=float((rep.estimate > 0).mean()), outcome_counts=rep.outcome.value_counts().to_dict(),
             median_ci_low=float(rep.ci_low.median()), median_ci_high=float(rep.ci_high.median()))
    return s
