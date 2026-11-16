"""Phylogenetic comparative toolkit for ANT-TWIN (no R needed at run time).

- Newick reading for the Economo et al. 2018 15K-species trees (plain Newick, tips `Genus.species[_subspecies]`, one tree per `;`).
- Phylogenetic covariance (VCV) among any subset of tips, computed without pruning the 15K tree.
- Matching our species keys (`genus_species`) to tree tips: exact -> subspecies tip -> recent genus transfer
  (same epithet, former genus; epithet gender endings tolerated) -> placement on an unused tip of the same genus
  (the owners' rule, Matte et al. 2025) -> unmatched. Every species carries how it was matched.
- GLS with a fixed correlation matrix (= nlme::gls(..., correlation = corSymm(fixed = TRUE), method = "ML")) and
  PGLS with Pagel's lambda estimated by maximum likelihood (= phylolm(model = "lambda")).
- Summaries across posterior trees.
Validated against R (nlme 3.1-171, phylolm 2.6.5) in the cuticle_replication stage.
"""
import re
import numpy as np
import pandas as pd
from scipy import optimize, stats

# ---------------------------------------------------------------- trees

_TOK = re.compile(r"\s*(\(|\)|,|;|:\s*[-+0-9.eE]+|\[[^\]]*\]|'[^']*'|[^\s(),:;\[\]]+)")


class Tree:
    """Rooted tree as arrays in preorder: parent[i] < i for every non-root node (root = 0)."""

    def __init__(self, parent, length, label):
        self.parent = np.asarray(parent, dtype=np.int64)
        self.length = np.asarray(length, dtype=float)
        self.label = label
        self.depth = np.zeros(len(parent))
        for i in range(1, len(parent)):  # preorder => parent depth already known
            self.depth[i] = self.depth[self.parent[i]] + self.length[i]
        is_parent = np.zeros(len(parent), bool); is_parent[self.parent[1:]] = True
        self.tips = [i for i in range(len(parent)) if not is_parent[i]]
        self.tip_labels = [label[i] for i in self.tips]  # in Newick order (= ape's tip.label order)
        self.tip_index = {label[i]: i for i in self.tips}

    @property
    def height(self):
        return float(self.depth[self.tips].max())

    def vcv(self, tips):
        """Shared root-to-MRCA path length for every pair of the given tip labels (order kept)."""
        idx = [self.tip_index[t] for t in tips]
        if len(set(idx)) != len(idx):
            raise ValueError("duplicate tips")
        n = len(idx); V = np.zeros((n, n))
        groups = {node: [pos] for pos, node in enumerate(idx)}
        for node in range(len(self.parent) - 1, 0, -1):  # reverse preorder = children before parents
            g = groups.pop(node, None)
            if g is None:
                continue
            p = self.parent[node]
            have = groups.get(p)
            if have is None:
                groups[p] = g
            else:
                V[np.ix_(have, g)] = self.depth[p]; V[np.ix_(g, have)] = self.depth[p]
                have.extend(g)
        V[np.arange(n), np.arange(n)] = self.depth[idx]
        return V


def parse_newick(s):
    parent, length, label = [-1], [0.0], [None]
    stack, cur = [], 0
    for m in _TOK.finditer(s):
        t = m.group(1)
        if t == "(":
            stack.append(cur); parent.append(cur); length.append(0.0); label.append(None); cur = len(parent) - 1
        elif t == ",":
            parent.append(stack[-1]); length.append(0.0); label.append(None); cur = len(parent) - 1
        elif t == ")":
            cur = stack.pop()
        elif t == ";":
            break
        elif t.startswith(":"):
            length[cur] = float(t[1:])
        elif t.startswith("["):
            continue
        else:
            label[cur] = t.strip("'")
    return Tree(parent, length, label)


def read_trees(path, max_trees=None):
    txt = open(path, encoding="utf-8").read()
    parts = [p for p in txt.split(";") if p.strip()]
    if max_trees:
        parts = parts[:max_trees]
    return [parse_newick(p + ";") for p in parts]


# ---------------------------------------------------------------- names

def tip_key(label):
    """'Camponotus.gigas_borneensis' -> 'camponotus_gigas_borneensis'."""
    return str(label).replace(".", "_").lower()


# Genera split or renamed after the Economo et al. 2018 taxonomy: current genus -> former genera in the tree.
GENUS_TRANSFERS = {
    "dinomyrmex": ["camponotus"], "colobopsis": ["camponotus"],
    "holcoponera": ["gnamptogenys"], "poneracantha": ["gnamptogenys"], "stictoponera": ["gnamptogenys"],
    "lioponera": ["cerapachys"], "parasyscia": ["cerapachys"], "ooceraea": ["cerapachys"], "syscia": ["cerapachys"],
    "neocerapachys": ["cerapachys"], "eburopone": ["cerapachys"], "chrysapace": ["cerapachys"],
    "zasphinctus": ["sphinctomyrmex"], "fulakora": ["stigmatomma"],
    "mycetomoellerius": ["trachymyrmex"], "paratrachymyrmex": ["trachymyrmex"], "mycetophylax": ["cyphomyrmex"],
    "patagonomyrmex": ["pogonomyrmex"], "iberoformica": ["formica", "proformica"], "xerolitor": ["mycetosoritis"],
    "fisheropone": ["pachycondyla", "mesoponera"], "rasopone": ["pachycondyla"],
}


def _stem(ep):
    for suf in ("us", "um", "a", "is", "e"):
        if ep.endswith(suf) and len(ep) - len(suf) >= 3:
            return ep[: -len(suf)]
    return ep


def match_species(species, tree_labels, transfers=GENUS_TRANSFERS, place_in_genus=True):
    """Map species keys (genus_species) to tree tip labels. Returns DataFrame [species_key, tip, how].
    how: exact | subspecies_tip | genus_transfer | epithet_gender | genus_placed | unmatched.
    Placement on an unused congeneric tip follows Matte et al. 2025 (closest unused tip in Newick order to an
    already used congener, else the first one); placed species are flagged so they can be dropped in robustness checks."""
    labels = list(tree_labels)
    by_key = {tip_key(l): l for l in labels}
    order = {l: i for i, l in enumerate(labels)}
    by_genus, by_gen_stem = {}, {}
    for l in labels:
        k = tip_key(l).split("_")
        by_genus.setdefault(k[0], []).append(l)
        if len(k) >= 2:
            by_gen_stem.setdefault((k[0], _stem(k[1])), []).append(l)
    rows, used = {}, set()

    def take(sp, tip, how):
        rows[sp] = (tip, how); used.add(tip)

    species = list(dict.fromkeys(species))
    for sp in species:  # pass 1: exact
        if sp in by_key and by_key[sp] not in used:
            take(sp, by_key[sp], "exact")
    for sp in species:  # pass 2: name-based fallbacks
        if sp in rows:
            continue
        g, ep = sp.split("_")[0], sp.split("_")[1]
        subs = sorted([l for k, l in by_key.items() if k.startswith(sp + "_") and l not in used], key=order.get)
        nominal = by_key.get(f"{sp}_{ep}")
        if subs:
            take(sp, nominal if nominal in subs else subs[0], "subspecies_tip"); continue
        cands = [l for og in transfers.get(g, []) for l in by_gen_stem.get((og, _stem(ep)), []) if l not in used]
        cands = [l for l in cands if len(tip_key(l).split("_")) == 2] or cands
        if cands:
            take(sp, cands[0], "genus_transfer"); continue
        cands = [l for l in by_gen_stem.get((g, _stem(ep)), []) if l not in used and len(tip_key(l).split("_")) == 2]
        if cands:
            take(sp, cands[0], "epithet_gender"); continue
    if place_in_genus:  # pass 3: unused congeneric tip
        species_set = set(species)
        for sp in species:
            if sp in rows:
                continue
            g = sp.split("_")[0]
            free = [l for l in by_genus.get(g, []) if l not in used and tip_key(l) not in species_set]
            if not free:
                continue
            taken = [order[t] for t, _ in rows.values() if tip_key(t).split("_")[0] == g]
            pick = min(free, key=lambda l: min(abs(order[l] - j) for j in taken)) if taken else min(free, key=order.get)
            take(sp, pick, "genus_placed")
    out = pd.DataFrame([{"species_key": sp, "tip": rows.get(sp, (None, "unmatched"))[0],
                         "how": rows.get(sp, (None, "unmatched"))[1]} for sp in species])
    return out


# ---------------------------------------------------------------- models

def design(df, numeric, categorical=None):
    """Intercept + numeric columns + treatment dummies. categorical: {col: [reference, level2, ...]}."""
    cols, names = [np.ones(len(df))], ["(Intercept)"]
    for c in numeric:
        cols.append(df[c].to_numpy(float)); names.append(c)
    for c, levels in (categorical or {}).items():
        for lv in levels[1:]:
            cols.append((df[c] == lv).to_numpy(float)); names.append(f"{c}{lv}")
    return np.column_stack(cols), names


def gls(y, X, C, names=None):
    """ML GLS with a fixed residual correlation/covariance matrix C (scale is estimated).
    Coefficient SEs use sigma^2 = RSS/(n-p) as nlme::gls and phylolm do; t-tests with n-p df."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, p = X.shape
    L = np.linalg.cholesky(C)
    Xs = np.linalg.solve(L, X); ys = np.linalg.solve(L, y)
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    r = ys - Xs @ beta; rss = float(r @ r)
    logdet = 2 * np.log(np.diag(L)).sum()
    loglik = -0.5 * (n * (np.log(2 * np.pi) + np.log(rss / n) + 1) + logdet)
    s2 = rss / (n - p)
    cov = s2 * np.linalg.inv(Xs.T @ Xs)
    se = np.sqrt(np.diag(cov)); tval = beta / se; df = n - p
    pval = 2 * stats.t.sf(np.abs(tval), df); q = stats.t.ppf(0.975, df)
    names = names or [f"x{i}" for i in range(p)]
    coef = pd.DataFrame({"term": names, "estimate": beta, "se": se, "t": tval, "p": pval,
                         "ci_low": beta - q * se, "ci_high": beta + q * se})
    return {"coef": coef, "n": n, "df": df, "logLik": loglik, "sigma2_ml": rss / n, "sigma2": s2}


def lambda_matrix(V, lam):
    """Pagel's lambda: off-diagonal (shared history) scaled by lambda, tip heights kept."""
    d = np.diag(V)
    return lam * V + (1 - lam) * np.diag(d)


def pgls_lambda(y, X, V, names=None, lam=None, bounds=(1e-7, 1.0)):
    """PGLS with Pagel's lambda (ML) unless lam is given. V is the phylogenetic VCV (any scale)."""
    V = V / np.mean(np.diag(V))
    if lam is not None:
        res = gls(y, X, lambda_matrix(V, lam), names); res["lambda"] = float(lam); return res
    f = lambda l: -gls(y, X, lambda_matrix(V, l), names)["logLik"]
    grid = np.linspace(bounds[0], bounds[1], 21)
    vals = [f(g) for g in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    opt = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    lam_hat = float(opt.x) if opt.fun <= vals[i] else float(grid[i])
    res = gls(y, X, lambda_matrix(V, lam_hat), names)
    res["lambda"] = lam_hat
    # profile-likelihood 95% CI for lambda (chi2_1 / 2 = 1.92)
    cut = res["logLik"] - stats.chi2.ppf(0.95, 1) / 2
    g = lambda l: -f(l) - cut
    res["lambda_ci"] = [float(optimize.brentq(g, bounds[0], lam_hat)) if g(bounds[0]) < 0 else float(bounds[0]),
                        float(optimize.brentq(g, lam_hat, bounds[1])) if g(bounds[1]) < 0 else float(bounds[1])]
    res["logLik_lambda0"] = -f(bounds[0]); res["logLik_lambda1"] = -f(bounds[1])
    return res


def ols(y, X, names=None):
    return gls(y, X, np.eye(len(y)), names)


def term(res, name):
    """One coefficient row as a flat dict."""
    r = res["coef"].set_index("term").loc[name]
    out = {k: float(r[k]) for k in ("estimate", "se", "t", "p", "ci_low", "ci_high")}
    out.update(n=res["n"], df=res["df"], logLik=res["logLik"], **{k: res[k] for k in ("lambda",) if k in res})
    if "lambda_ci" in res:
        out["lambda_ci_low"], out["lambda_ci_high"] = res["lambda_ci"]
    return out


def summarize_over_trees(per_tree, draws=2000, seed=1):
    """per_tree: DataFrame with estimate, se, df, p (one row per tree). Pooled 95% interval = quantiles of the
    mixture of per-tree t sampling distributions (tree + sampling uncertainty)."""
    rng = np.random.default_rng(seed)
    mix = np.concatenate([r.estimate + r.se * rng.standard_t(r.df, draws) for r in per_tree.itertuples()])
    return {"n_trees": int(len(per_tree)), "estimate_median": float(per_tree.estimate.median()),
            "estimate_min": float(per_tree.estimate.min()), "estimate_max": float(per_tree.estimate.max()),
            "pooled_ci_low": float(np.quantile(mix, 0.025)), "pooled_ci_high": float(np.quantile(mix, 0.975)),
            "share_p_lt_0.05": float((per_tree.p < 0.05).mean()), "p_max": float(per_tree.p.max()),
            "lambda_median": float(per_tree["lambda"].median()) if "lambda" in per_tree else None,
            "n_species_min": int(per_tree.n.min()), "n_species_max": int(per_tree.n.max())}
