"""Figures for the ANT-TWIN preprint draft: Figs 2-4, Fig. S1 (pipeline schematic) and the accuracy panels
of Fig. 1 (draw_accuracy, called by make_image_figures.py, which renders the image-based Fig. 1).

Reads only local copies of committed / R2 outputs; changes nothing upstream.
  research/ant-twin/results/brain_table_final_v1.csv          frozen brain table (689 specimens)
  research/ant-twin/work/seg_v3/fold_k_val_per_case.json        seg v3 held-out cases (copy of R2 runs/20261114-212341_*,
                                                                 runs/20261114-215207_* outputs/fold_k/)
  research/ant-twin/work/robustness/run_outputs/                primary run outputs (copy of R2 runs/20261115-100746_*/outputs)
  research/ant-twin/work/robustness/{r8,r2,r3x}.json, me/       robustness outputs (copy of R2 analysis/robustness_v1, r3_extension_v1)
  research/ant-twin/preprint/tables/supp_checks.json            draft-only checks (tables/supp_checks.py)
  research/ant-twin/preprint/tables/species_soft_adaptive.csv   species soft tissue / cuticle fraction (tables/supp_checks.py)
Run:  uv run --no-project --python 3.13 --with pandas,numpy,scipy,matplotlib python make_figures.py
Output: figS1, fig2..fig4 as PNG (300 dpi) and PDF next to this script (fig1 comes from make_image_figures.py).
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

ROOT = pathlib.Path("C:/projects/dsc-biology/research/ant-twin")
OUT = pathlib.Path(__file__).resolve().parent
RES, WORK = ROOT / "results", ROOT / "work"
RO = WORK / "robustness" / "run_outputs"
SUPP = ROOT / "preprint" / "tables"
SESOI = 0.02

# ---------------------------------------------------------------- style (colour-blind-safe reference palette)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3 (validated all-pairs)
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
OUTCOME_COL = {"support_H2": BLUE, "undecided": MUTED, "equivalence_H0": AQUA, "support_H1": ORANGE, "small_but_nonzero": INK2}
OUTCOME_LAB = {"support_H2": "support H2 (95% CI > 0)", "undecided": "undecided",
               "equivalence_H0": "equivalence (90% CI within ±0.02)"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "legend.frameon": False})
MM = 1 / 25.4


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def classify(est, lo95, hi95, lo90, hi90):
    """Prereg sec. 7 decision rules."""
    excl0 = lo95 > 0 or hi95 < 0
    if excl0 and -SESOI < lo95 and hi95 < SESOI:
        return "small_but_nonzero"
    if hi95 < 0:
        return "support_H1"
    if lo95 > 0:
        return "support_H2"
    if -SESOI < lo90 and hi90 < SESOI:
        return "equivalence_H0"
    return "undecided"


def ci90_from_se(est, se, df_):
    q = stats.t.ppf(0.95, df_)
    return est - q * se, est + q * se


# ---------------------------------------------------------------- load
prim = json.loads((RO / "primary.json").read_text())
haller = json.loads((RO / "haller.json").read_text())
r8 = json.loads((WORK / "robustness/r8.json").read_text())
r2 = json.loads((WORK / "robustness/r2.json").read_text())
r3x = json.loads((WORK / "robustness/r3x.json").read_text())
supp = json.loads((SUPP / "supp_checks.json").read_text())
brms = pd.read_csv(WORK / "robustness/me/brms_summary_m_main_iter4000.csv").set_index("variable")
mcmc = pd.read_csv(WORK / "robustness/me/mcmcglmm_summary_m_main.csv").set_index("term")
pdat = pd.read_csv(RO / "primary_data.csv", dtype={"vox_class": str})
btab = pd.read_csv(RES / "brain_table_final_v1.csv")
soft = pd.read_csv(SUPP / "species_soft_adaptive.csv")
seg = pd.DataFrame([{"fold": f, "case": k, **v} for f in range(5)
                    for k, v in json.loads((WORK / f"seg_v3/fold_{f}_val_per_case.json").read_text()).items()])
assert len(pdat) == 270 and len(btab) == 689 and len(soft) == 235
assert seg.groupby("source").size().to_dict() == {"AS": 24, "AS2": 530, "TK": 117}


# ================================================================ Fig S1: pipeline schematic (was Fig. 1a in draft v1)
# The accuracy plots (old Fig. 1b-c) are drawn by draw_accuracy() into the new image-based Fig. 1 (d-e) of
# make_image_figures.py; the schematic moved to the supplement.
def figS1():
    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    ax = fig.add_subplot(111); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
    draw_schematic(ax)
    save(fig, "figS1_pipeline_schematic")


def draw_schematic(ax):

    def box(x, y, w, h, title, body, fc="#f3f7fd", ec=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=6.0, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.9, body, ha="center", va="top", fontsize=5.5, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8,
                                                                      shrinkA=0, shrinkB=0, mutation_scale=7))
    # row 1: measurement chain
    w, h, y1 = 14.6, 15.5, 26.5
    xs = [0.5, 17.2, 33.9, 50.6, 67.3, 84.0]
    items = [("AntScan scans", "synchrotron (KIT)\n818 colony workers\n352 species with\nknown colony size"),
             ("Head finder", "811 low-res stacks\nnnU-Net 3d\nlocates the brain"),
             ("Head box", "full-resolution\ncrop around\nthe brain"),
             ("Segmentation", "nnU-Net 2d\n5-fold ensemble;\nheld-out Dice\n0.962-0.973"),
             ("Vision-agent QC", "every mask: box\non head, mask good,\nbrain fully in box"),
             ("Brain table", "689 workers\n325 species\nfrozen, SHA-256\nICC(1) = 0.90")]
    for x, (t, b) in zip(xs, items):
        box(x, y1, w, h, t, b)
    for a, b_ in zip(xs[:-1], xs[1:]):
        arrow(a + w + 0.3, y1 + h / 2, b_ - 0.3, y1 + h / 2)
    ax.text(xs[3] + w / 2, y1 - 1.0, "trained on Toulkeridou et al. 2023 (117 stacks, human labels)\n+ 24 bootstrap reference masks + 530 agent-approved AntScan brains",
            ha="center", va="top", fontsize=5.4, color=INK2, style="italic")
    # row 2: covariates and model
    y2, h2 = 1.0, 15.5
    box(0.5, y2, 17.5, h2, "Body volume", "S1 of Matte et al. 2025\nwhere measured, else\nour body masks\n(ratio to S1: 1.04)", fc="#fbfbfa", ec=MUTED)
    box(20.0, y2, 17.5, h2, "Soft tissue (R3)", "body - cuticle\nS1 (154 species) +\nowners' program on\n141 workers (81 spp.)", fc="#fef4ef", ec=ORANGE)
    box(39.5, y2, 17.5, h2, "Colony size", "species means from\nMatte et al. 2025 S3;\nBell-Roberts et al.\n2024 as fallback", fc="#fbfbfa", ec=MUTED)
    box(59.0, y2, 17.5, h2, "Phylogeny", "Economo et al. 2018\nMCC + posterior trees;\nBorowiec et al. 2025\n(check R2)", fc="#fbfbfa", ec=MUTED)
    box(80.5, y2, 19.0, h2, "Preregistered PGLS", "log10 brain ~ log10 body\n+ log10 colony + stain\n+ scan step\n270 species", fc="#eef8f3", ec=AQUA)
    arrow(xs[5] + w / 2, y1 - 0.4, 90.0, y2 + h2 + 0.4)
    arrow(76.5 + 0.3, y2 + h2 / 2, 80.5 - 0.3, y2 + h2 / 2)


def draw_accuracy(axes, letters=("d", "e"), compact=False, panel_x=-0.12):
    """Held-out accuracy of seg v3 by reference type (Dice, |relative volume error|); used by the image-based Fig. 1."""
    order = [("TK", "human labels\n(Toulkeridou\n2023)"), ("AS", "AntScan\nbootstrap\nreference masks"),
             ("AS2", "AntScan\nagent-approved\nmasks")]
    if compact:
        order = [("TK", "Human labels\n(Toulkeridou\n2023)"), ("AS", "AntScan\nbootstrap\nreference\nmasks"), ("AS2", "AntScan\nagent-\napproved")]
    colors = {"TK": BLUE, "AS": ORANGE, "AS2": AQUA}
    rng = np.random.default_rng(3)
    n_low = int((seg.dice < 0.80).sum()); n_hi = int((seg.rel_vol_err.abs() > 0.25).sum())
    for j, (metric, ylab, letter) in enumerate([("dice", "Dice (held-out folds)", letters[0]),
                                                ("abs_err", "|relative volume error| (%)", letters[1])]):
        a = axes[j]; panel(a, letter, x=panel_x)
        for i, (src, lab) in enumerate(order):
            s = seg[seg.source == src]
            v = s.dice.to_numpy() if metric == "dice" else 100 * s.rel_vol_err.abs().to_numpy()
            a.boxplot(v, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor="none", edgecolor=INK2, lw=0.7), medianprops=dict(color=INK, lw=1.1),
                      whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
            a.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=6, color=colors[src], alpha=0.55, lw=0, zorder=3)
            med = np.median(v)
            a.text(i + 0.30, med, f"{med:.3f}" if metric == "dice" else f"{med:.1f}%", fontsize=6.3, va="center", color=INK)
        a.set_xticks(range(3)); a.set_xticklabels([f"{lab}\n(n = {int((seg.source == s_).sum())})" for s_, lab in order])
        a.set_ylabel(ylab); a.grid(axis="x", visible=False)
        if metric == "dice":
            a.set_ylim(0.80, 1.0)
            a.text(0.02, 0.03, f"y-axis clipped at 0.80;\n{n_low} of 671 cases below not shown", transform=a.transAxes,
                   fontsize=5.8, color=INK2, va="bottom")
        else:
            a.set_ylim(0, 25)
            a.text(0.02, 0.98, f"y-axis clipped at 25%;\n{n_hi} of 671 cases above not shown", transform=a.transAxes,
                   fontsize=5.8, color=INK2, va="top")
        if compact:
            a.tick_params(axis="x", labelsize=6.0)


# ================================================================ Fig 2: allometry + partial colony effect
def fig2():
    d = pdat.copy()
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 76 * MM), gridspec_kw=dict(wspace=0.42))
    # (a) Haller's rule, brain adjusted for stain and scan-step class with the gate-3 PGLS coefficients
    hc = {c["term"]: c["estimate"] for c in haller["coef"]}
    adj = (d.log10_brain - hc["stainiodine"] * (d.stain == "iodine") - hc["vox_class1.22"] * (d.vox_class == "1.22")
           - hc["vox_class6.11"] * (d.vox_class == "6.11"))
    a = axs[0]; panel(a, "a")
    order = np.argsort(d.log10_colony_any.to_numpy())
    sc = a.scatter(d.log10_body.to_numpy()[order], adj.to_numpy()[order], c=d.log10_colony_any.to_numpy()[order], cmap=SEQ,
                   vmin=0.8, vmax=7.0, s=16, edgecolor=SURF, linewidth=0.35, zorder=3)
    xs = np.linspace(d.log10_body.min(), d.log10_body.max(), 50)
    b1 = haller["slope"]; c0 = adj.mean() - b1 * d.log10_body.mean()
    a.plot(xs, c0 + b1 * xs, color=INK, lw=1.2, zorder=4)
    a.text(0.03, 0.97, f"PGLS slope {b1:.3f} [{haller['ci_low']:.3f}; {haller['ci_high']:.3f}]\n"
                       f"λ = {haller['lambda']:.2f}, R² (OLS) = {haller['r2_ols_same_terms']:.2f}, n = {haller['n_species']} species",
           transform=a.transAxes, va="top", fontsize=6.5)
    a.set_xlabel("log10 body volume (mm³)"); a.set_ylabel("log10 brain volume (mm³), adjusted")
    a.set_ylim(adj.min() - 0.15, adj.max() + 0.45)
    cb = fig.colorbar(sc, ax=a, fraction=0.045, pad=0.02); cb.set_label("log10 colony size", fontsize=6.8)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6.3)
    # (b) partial (component + residual) plot of the colony term from the preregistered MCC fit
    co = {c["term"]: c["estimate"] for c in prim["coef_mcc"]}
    m = prim["mcc"]
    fitted = (co["(Intercept)"] + co["log10_body"] * d.log10_body + co["log10_colony_any"] * d.log10_colony_any
              + co["stainiodine"] * (d.stain == "iodine") + co["vox_class1.22"] * (d.vox_class == "1.22")
              + co["vox_class6.11"] * (d.vox_class == "6.11"))
    x = d.log10_colony_any.to_numpy(); b2 = co["log10_colony_any"]
    pr = (d.log10_brain - fitted).to_numpy() + b2 * x
    pr = pr - pr.mean()
    a = axs[1]; panel(a, "b")
    a.scatter(x, pr, s=12, color=MUTED, alpha=0.75, lw=0, zorder=3)
    xg = np.linspace(x.min(), x.max(), 60); xm = x.mean(); q = stats.t.ppf(0.975, m["df"])
    a.fill_between(xg, (b2 - q * m["se"]) * (xg - xm), (b2 + q * m["se"]) * (xg - xm), color=BLUE, alpha=0.18, lw=0, zorder=2)
    a.plot(xg, b2 * (xg - xm), color=BLUE, lw=1.6, zorder=4)
    a.axhline(0, color=INK2, lw=0.5, ls=(0, (3, 3)), zorder=1)
    a.text(0.03, 0.97, f"β₂ = +{b2:.3f} [{m['ci_low']:.3f}; {m['ci_high']:.3f}], p = {m['p']:.4f}\n"
                       f"≈ +{100 * (10 ** b2 - 1):.1f}% brain volume per tenfold colony size\n"
                       f"MCC tree, λ = {m['lambda']:.2f}, n = {m['n']} species → support H2",
           transform=a.transAxes, va="top", fontsize=6.5)
    a.set_xlabel("log10 colony size (workers)"); a.set_ylabel("partial residual, log10 brain volume")
    a.set_ylim(pr.min() - 0.08, pr.max() + 0.32)
    save(fig, "fig2_allometry_colony")


# ================================================================ Fig 3: forest plot of beta2
def fig3_rows():
    R = prim["robustness"]; rows = []

    def add(group, label, est, lo, hi, lo90, hi90, n, prereg=True, outcome=None):
        rows.append(dict(group=group, label=label, est=est, lo=lo, hi=hi, n=n, prereg=prereg,
                         outcome=outcome or classify(est, lo, hi, lo90, hi90)))

    def add_fit(group, label, r, prereg=True, n=None):
        df_ = (r.get("n") or n) - 6
        lo90, hi90 = r["ci90"] if "ci90" in r else ci90_from_se(r["estimate"], r["se"], df_)
        add(group, label, r["estimate"], r["ci_low"], r["ci_high"], lo90, hi90, n or r["n"], prereg, r.get("outcome"))

    def add_pool(group, label, s, n, prereg=True):
        add(group, label, s["estimate_median"], s["pooled_ci_low"], s["pooled_ci_high"], s["pooled_ci90_low"],
            s["pooled_ci90_high"], n, prereg, s.get("outcome"))

    G = "Primary model"
    add_fit(G, "MCC tree (primary)", prim["mcc"])
    add_pool(G, "100 posterior trees (pooled)", prim["posterior"], 270)
    b = brms.loc["b_log10_colony"]
    add(G, "Measurement-error model, brms", b["mean"], b["2.5%"], b["97.5%"], b["5%"], b["95%"], "442 / 270")
    mc = mcmc.loc["log10_colony"]
    add(G, "Measurement-error model, MCMCglmm", mc["mean"], mc["q025"], mc["q975"], np.nan, np.nan, "442 / 270",
        outcome=classify(mc["mean"], mc["q025"], mc["q975"], -1, 1))
    G = "Phylogeny"
    add_pool(G, "R1 FBD trees (6, primary run)", R["R1"]["result"], 270)
    s200 = supp["R1_fbd_200"]
    add_pool(G, "R1 FBD trees (all 200; post hoc check)", s200, 270, prereg=False)
    k = [k for k in r2 if k.startswith("129-158Ma_independent") and "272" in k and "| 100 resamples" in k][0]
    add_pool(G, "R2 Borowiec 2025 tree, 1 sp./genus", r2[k]["summary"], 117)
    add_fit(G, "R12 λ = 1 (Brownian)", R["R12"]["result"]["lambda_1"])
    ow = R["R12"]["result"]["owners_meanVCV_div200"]
    add_fit(G, "R12 owners' mean VCV (6 trees)", ow)
    add_fit(G, "R14 OLS, no phylogeny", R["R14"]["result"])
    G = "Species set"
    for sf, r in R["R4"]["result"].items():
        add_fit(G, f"R4 without {sf}", r)
    add_fit(G, "R5 unstained species only", R["R5"]["result"])
    add_fit(G, "R6 all workers incl. majors", R["R6"]["result"])
    add_fit(G, "R7 without cf./aff./nr. names", R["R7"]["result"])
    add_fit(G, "R8 without supercolonial / parasites", r8["mcc"])
    add_fit(G, "R11 without genus-placed tips", R["R11"]["result"])
    add_fit(G, "R13 + badly preserved, year covariate", R["R13"]["result"])
    G = "Colony-size measure"
    add_fit(G, "R9 Bell-Roberts 2024 colony size", R["R9"]["result"]["log10_colony_br"])
    add_fit(G, "R9 maximum instead of mean", R["R9"]["result"]["log10_colony_max_owners"])
    add_fit(G, "R10 genus-level colony size", R["R10"]["result"])
    G = "Size denominator"
    f = r3x["fits"]
    add_fit(G, "R3 soft tissue, S1 only", f["R3_prereg_S1_only"])
    add_fit(G, "R3x soft tissue, S1 + own (adaptive QC)", f["R3x_adaptive"], prereg=False)
    add_fit(G, "R3x soft tissue, strict QC", f["R3x_t110"], prereg=False)
    add_fit(G, "Body, same 235 species as R3x", f["R3x_adaptive_body_same_species"], prereg=False)
    lf = supp["like_for_like"]["all_235"]
    add_fit(G, "Body, same 235 spp., same instrument as soft", lf["body_same_instrument"], prereg=False)
    G = "Body-volume quality (post hoc checks)"
    bq = supp["body_quality"]
    add_fit(G, "Drop specimens with brain/body > 10%", bq["drop_brain_body_gt10pct"], prereg=False)
    add_fit(G, "Drop specimens with brain/body > 5%", bq["drop_brain_body_gt5pct"], prereg=False)
    add_fit(G, "Drop all autoflagged specimens", bq["drop_all_autoflagged"], prereg=False)
    add_fit(G, "Owners' S1 body volumes only", bq["S1_bodies_only"], prereg=False)
    add_fit(G, "Our body-mask volumes only", bq["own_v8_bodies_only"], prereg=False)
    return pd.DataFrame(rows)


def fig3():
    rows = fig3_rows()
    rows.to_csv(OUT / "fig3_data.csv", index=False)
    groups = list(dict.fromkeys(rows.group))
    ypos, labels, y, heads = [], [], 0, []
    for g in groups:
        heads.append((y, g)); y += 1
        for _ in range(int((rows.group == g).sum())):
            ypos.append(y); y += 1
        y += 0.4
    rows["y"] = ypos
    fig, ax = plt.subplots(figsize=(150 * MM, 200 * MM))
    ax.axvspan(-SESOI, SESOI, color="#f0efec", lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    ax.axvline(prim["mcc"]["estimate"], color=BLUE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for r in rows.itertuples():
        c = OUTCOME_COL[r.outcome]
        ax.plot([r.lo, r.hi], [r.y, r.y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
        ax.scatter([r.est], [r.y], s=26 if r.prereg else 24, marker="o" if r.prereg else "D",
                   facecolor=c if r.prereg else SURF, edgecolor=c, linewidth=1.1, zorder=3)
        ax.text(0.079, r.y, f"{r.est:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]", va="center", fontsize=6.2, color=INK)
        ax.text(0.1135, r.y, str(r.n), va="center", fontsize=6.2, color=INK2)
    for yh, g in heads:
        ax.text(-0.0545, yh, g, va="center", fontsize=7, fontweight="bold", color=INK)
    ax.set_yticks(rows.y); ax.set_yticklabels(rows.label, fontsize=6.6)
    ax.set_ylim(y - 0.2, -1.2)
    ax.set_xlim(-0.025, 0.075)
    ax.text(0.079, -0.9, "β₂ [95% interval]", fontsize=6.4, fontweight="bold")
    ax.text(0.1135, -0.9, "n", fontsize=6.4, fontweight="bold")
    ax.text(-0.0195, -0.9, "SESOI band ±0.02", fontsize=6.0, color=INK2, ha="left")
    ax.set_xlabel("β₂: slope of log10 brain volume on log10 colony size (body size, stain, scan step held constant)")
    ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    hand = [Line2D([], [], color=OUTCOME_COL[o], marker="o", lw=1.4, ms=4.5, label=OUTCOME_LAB[o])
            for o in ("support_H2", "undecided", "equivalence_H0")]
    hand += [Line2D([], [], color=INK2, marker="o", lw=0, ms=4.5, label="preregistered"),
             Line2D([], [], color=INK2, marker="D", mfc=SURF, lw=0, ms=4.2, label="exploratory / post hoc check")]
    ax.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.55, -0.07), fontsize=6.3, ncol=3, handlelength=1.6,
              frameon=False)
    save(fig, "fig3_forest_beta2")


# ================================================================ Fig 4: size denominators, cuticle, body-volume quality
def fig4():
    f = r3x["fits"]; lf = supp["like_for_like"]
    sets = [("154 S1\nspecies", [("body, primary table", supp["S1_154_body"], BLUE, "o"),
                                        ("body, same instrument", lf["S1_154"]["body_same_instrument"], INK2, "D"),
                                        ("soft tissue", f["R3_prereg_S1_only"], ORANGE, "s")]),
            ("235 species\n(S1 + own)", [("body, primary table", f["R3x_adaptive_body_same_species"], BLUE, "o"),
                                                  ("body, same instrument", lf["all_235"]["body_same_instrument"], INK2, "D"),
                                                  ("soft tissue", f["R3x_adaptive"], ORANGE, "s")])]
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 66 * MM), gridspec_kw=dict(width_ratios=[1.0, 1.05, 0.85], wspace=0.42))
    a = axs[0]; panel(a, "a", x=-0.2)
    a.axhspan(-SESOI, SESOI, color="#f0efec", lw=0, zorder=0); a.axhline(0, color=INK2, lw=0.7, zorder=1)
    for i, (lab, items) in enumerate(sets):
        for j, (nm, r, col, mk) in enumerate(items):
            x = i + (j - 1) * 0.24
            a.plot([x, x], [r["ci_low"], r["ci_high"]], color=col, lw=1.5, solid_capstyle="round", zorder=2)
            a.scatter([x], [r["estimate"]], color=col, marker=mk, s=24, edgecolor=SURF, linewidth=0.6, zorder=3)
            a.text(x, r["ci_high"] + 0.002, f"{r['estimate']:+.3f}", fontsize=5.6, ha="center", va="bottom", color=INK, rotation=90)
    a.set_xticks(range(2)); a.set_xticklabels([s_[0] for s_ in sets], fontsize=6.4)
    a.set_xlim(-0.55, 1.55); a.set_ylim(-0.03, 0.085); a.set_ylabel("β₂ (log10 colony), same species"); a.grid(axis="x", visible=False)
    a.legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=1.5, ms=4, label="body, primary table (S1 or our body mask)"),
                      Line2D([], [], color=INK2, marker="D", lw=1.5, ms=3.6, label="body, same instrument and specimens as soft"),
                      Line2D([], [], color=ORANGE, marker="s", lw=1.5, ms=4, label="soft tissue (body − cuticle)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=5.8, ncol=1)
    # (b) cuticle fraction (same-specimen body and cuticle) vs colony size
    a = axs[1]; panel(a, "b", x=-0.2)
    for src, col, mk, lab in (("S1", BLUE, "o", "S1 (Matte et al. 2025)"), ("ours", ORANGE, "s", "owners' program, this study")):
        s_ = soft[soft.src == src]
        a.scatter(s_.log10_colony_any, 100 * s_.cuticle_frac_same, s=10, color=col, marker=mk, alpha=0.75, lw=0,
                  label=f"{lab} (n = {len(s_)})", zorder=3)
    x, yv = soft.log10_colony_any.to_numpy(), 100 * soft.cuticle_frac_same.to_numpy()
    bb = np.polyfit(x, yv, 1); xg = np.linspace(x.min(), x.max(), 20)
    a.plot(xg, np.polyval(bb, xg), color=INK, lw=1.0, zorder=4)
    a.text(0.97, 0.97, f"r(log10 colony, log10 soft/body) = +{lf['all_235']['corr_colony_log_soft_over_body']:.2f}\n"
                       f"n = {len(soft)} species; line: OLS, descriptive",
           transform=a.transAxes, ha="right", va="top", fontsize=5.9)
    a.set_xlabel("log10 colony size (workers)"); a.set_ylabel("cuticle / body volume (%)")
    a.set_ylim(0, max(50, float(np.nanmax(yv)) + 11))
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=5.8, markerscale=1.3)
    # (c) brain/body fraction by body-volume source (diagnostic for body-volume errors)
    a = axs[2]; panel(a, "c", x=-0.3)
    sp = pd.read_csv(RO / "specimens_table.csv", usecols=["main_ok", "log10_body", "brain_um3", "body_um3", "body_source"])
    sp = sp[sp.main_ok & sp.log10_body.notna()].assign(bf=lambda d: 100 * d.brain_um3 / d.body_um3)
    rng = np.random.default_rng(5)
    for i, (src, col) in enumerate((("S1", BLUE), ("v6", ORANGE))):
        v = sp[sp.body_source == src].bf.to_numpy()
        a.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=col, alpha=0.6, lw=0, zorder=3)
        a.boxplot(v, positions=[i], widths=0.5, showfliers=False, boxprops=dict(color=INK2, lw=0.7),
                  medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
        a.text(i, 700, f"n = {len(v)}\n> 10%: {(v > 10).sum()}", ha="center", va="top", fontsize=5.8, color=INK2)
    a.set_yscale("log"); a.set_ylim(0.08, 1000)
    a.axhline(10, color=INK2, lw=0.6, ls=(0, (3, 3))); a.text(1.5, 10.5, "10%", fontsize=5.8, color=INK2, ha="right", va="bottom")
    a.set_xticks([0, 1]); a.set_xticklabels(["owners' S1\nbodies", "our body\nmasks"], fontsize=6.4); a.set_xlim(-0.55, 1.55)
    a.set_ylabel("brain / body volume (%), per worker"); a.grid(axis="x", visible=False)
    save(fig, "fig4_denominators_cuticle")


if __name__ == "__main__":
    # Fig. 1 (images + accuracy d-e) is written by make_image_figures.py; the schematic is Fig. S1.
    figS1(); fig2(); fig3(); fig4()
