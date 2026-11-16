"""Image figures for the ANT-TWIN preprint: main Fig. 1 (from scan to brain volume) and Supplementary Figs S2-S4.

Every panel is rendered from data stored in R2 (bucket dsc-biology, prefix ant-twin/); nothing upstream is changed.
Selections are rule-based with fixed seeds and are written, with the R2 path of every input, to
image_data/image_figures_selection.json next to this script.

  Fig. 1   (a) low-res whole specimen (cache/lowres/<key>.npz) with the full-resolution head box (box_zyx of the box npz)
           (b) 3D surface of the same specimen's mask inside its head box
           (c) gallery of 8 specimens: head-box slice + frozen seg v3 ensemble mask (cache/boxes_*/, cache/brain_masks_final*/)
           (d, e) held-out accuracy of seg v3 (drawn by make_figures.draw_accuracy, same data as before)
  Fig. S1  pipeline schematic (written by make_figures.py)
  Fig. S2  seg v3 held-out fold model vs the human label of Toulkeridou et al. (2023) on 4 TK cases
           (runs/20261113-181636_*/outputs/nnUNet_raw/, fold checkpoints of runs/20261114-212341_* and -215207_*)
  Fig. S3  body/cuticle segmentation by the owners' program: one case accepted and one rejected by the adaptive QC rule
           (analysis/r3_extension_v1/cuticle_final.jsonl, runs/<cuticle run>/outputs/qc/<key>.jpg and slices/<key>.npz)
  Fig. S4  masks rejected by the vision-agent QC in the final round (meta/brains_v5_auto_review/, the QC sheets
           review/brains_v5/qc/<key>.jpg that the agent saw = runs/<seg-final-v5 run>/outputs/qc/<key>.jpg)

Run from C:/projects/ant-twin/src (for r2.py) or set ANT_TWIN_SRC:
  render (default):   uv run --no-project --python 3.13 --with boto3 --with numpy --with scipy --with pandas \
                        --with pillow --with matplotlib --with scikit-image python make_image_figures.py
  TK inference (once, CPU, ~1-1.5 h on 8+ cores; writes image_data/tk_heldout.json + .npz; run before render):
                      uv run --no-project --python 3.12 --with nnunetv2 --with nibabel --with boto3 \
                        python make_image_figures.py tk-infer
  upload figures to R2 ant-twin/preprint/v1/figures/:  ... python make_image_figures.py upload
Downloads are cached in $ANT_TWIN_FIG_CACHE (default: <tmp>/ant_twin_fig_cache).
"""
import io, os, sys, json, hashlib, pathlib, tempfile

ROOT = pathlib.Path("C:/projects/dsc-biology/research/ant-twin")
OUT = pathlib.Path(__file__).resolve().parent
DATA = OUT / "image_data"
CACHE = pathlib.Path(os.environ.get("ANT_TWIN_FIG_CACHE", pathlib.Path(tempfile.gettempdir()) / "ant_twin_fig_cache"))
sys.path.insert(0, os.environ.get("ANT_TWIN_SRC", "C:/projects/ant-twin/src"))
RO = ROOT / "work" / "robustness" / "run_outputs"          # local copy of R2 runs/20261115-100746_*/outputs
REVIEW = ROOT / "review"                                    # QC verdicts (git; mirrored in R2 meta/*_auto_review/)

FREEZE = "ant-twin/runs/20261115-093934_brain-freeze-v1-final-brain-table-seg-v3-ensembl_DESKTOP-7Q7S0N3/outputs/"
ITEMS = {"main": "ant-twin/meta/seg_final_items.json", "v5": "ant-twin/meta/seg_final_items_v5.json",
         "alt": "ant-twin/meta/seg_final_items_alt.json"}
ALT_RUNS = ("20261115-083646", "20261115-085137")            # seg-final alt runs write cache/brain_masks_final_alt/
TK_RAW = "ant-twin/runs/20261113-181636_tk-volumes-v2-skip-macos-appledouble-entries_6945dce09bf2/outputs/"
SEG_V3 = {f: ("ant-twin/runs/20261114-212341_seg-v3-5-fold-nnu-net-2d-on-tk-all-agent-approve_9f0551f91543/outputs/"
              if f < 3 else "ant-twin/runs/20261114-215207_seg-v3-folds-3-4_5d5eb7649d55/outputs/") + f"fold_{f}/"
          for f in range(5)}
CUT_FINAL = "ant-twin/analysis/r3_extension_v1/cuticle_final.jsonl"
SEED_GALLERY, SEED_CUTICLE, SEED_QC = 20261116, 20261116, 20261116
UPLOAD_PREFIX = "ant-twin/preprint/v1/figures/"

_client = None


def r2c():
    global _client
    if _client is None:
        import r2
        _client = r2.client()
    return _client


def fetch(key):
    """Local cached copy of an R2 object (bucket dsc-biology)."""
    import r2
    dst = CACHE / key
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        r2c().download_file(r2.BUCKET, key, str(tmp)); tmp.replace(dst)
    return dst


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================ TK held-out inference (nnU-Net env only)
TK_CASES = None  # chosen in tk_select()


def tk_select():
    """Held-out TK cases of seg v3 at the 10th, 50th and 90th percentile of per-case Dice, plus the minimum.
    Per-case Dice of the official held-out validation (val_per_case.json of each fold, R2 seg v3 runs)."""
    import numpy as np
    rows = []
    for f in range(5):
        for case, v in json.loads(fetch(SEG_V3[f] + "val_per_case.json").read_text()).items():
            if v["source"] == "TK":
                rows.append(dict(case=case, fold=f, species=v["species"], dice=v["dice"], rel_vol_err=v["rel_vol_err"]))
    assert len(rows) == 117
    d = np.array([r["dice"] for r in rows]); pick = []
    for lab, q in (("10th percentile", 0.10), ("median", 0.50), ("90th percentile", 0.90)):
        i = int(np.argmin(np.abs(d - np.quantile(d, q)) + 1e-9 * np.arange(len(d))))
        pick.append({**rows[i], "role": lab})
    i = int(np.argmin(d)); pick.append({**rows[i], "role": "minimum"})
    return pick, {"n": len(rows), "median": float(np.median(d)), "q10": float(np.quantile(d, 0.1)),
                  "q90": float(np.quantile(d, 0.9)), "min": float(d.min()), "n_below_0.90": int((d < 0.90).sum())}


def tk_infer(full_volume=("TK_057", "TK_063")):
    """Run the held-out fold model of seg v3 (the fold whose validation set contains the case) on the TK image, exactly
    as nnU-Net validation does (2d, mirroring on, tile step 0.5, Gaussian). The 2d model predicts slice by slice, so
    the displayed slice is predicted on its own; selected cases are also predicted in full to re-check the Dice."""
    import numpy as np, nibabel as nib, torch, shutil
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    pick, stats = tk_select()
    M = CACHE / "tk_model"; M.mkdir(parents=True, exist_ok=True)
    for fn in ("plans.json", "dataset.json"):
        shutil.copy(fetch(SEG_V3[0] + fn), M / fn)
    out = {"rule": "held-out TK cases of seg v3 at the 10th/50th/90th percentile of per-case Dice and the minimum",
           "stats_117_tk_cases": stats, "cases": []}
    arrs = {}
    for r in pick:
        case, f = r["case"], r["fold"]
        assert json.loads(fetch(SEG_V3[f] + "plans.json").read_text()) == json.loads((M / "plans.json").read_text())
        ck = M / f"fold_{f}" / "checkpoint_final.pth"
        if not ck.exists():
            ck.parent.mkdir(exist_ok=True); shutil.copy(fetch(SEG_V3[f] + "checkpoint_final.pth"), ck)
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=False,
                            device=torch.device("cpu"), verbose=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(M), use_folds=(f,), checkpoint_name="checkpoint_final.pth")
        ik, lk = TK_RAW + f"nnUNet_raw/Dataset501_AntBrainTK/imagesTr/{case}_0000.nii.gz", TK_RAW + f"nnUNet_raw/Dataset501_AntBrainTK/labelsTr/{case}.nii.gz"
        img = np.asarray(nib.load(str(fetch(ik))).dataobj).transpose(2, 1, 0).astype(np.float32)  # as SimpleITK: (z, y, x)
        lab = np.asarray(nib.load(str(fetch(lk))).dataobj).transpose(2, 1, 0) > 0
        k = int(np.argmax(lab.sum((1, 2))))                                                           # largest label slice
        pr = p.predict_single_npy_array(img[None, k:k + 1], {"spacing": [1.0, 1.0, 1.0]}, None, None, False)[0] > 0
        rec = {**r, "slice": k, "n_slices": int(img.shape[0]), "image_r2": ik, "label_r2": lk,
               "checkpoint_r2": SEG_V3[f] + "checkpoint_final.pth",
               "slice_dice": round(float(2 * (pr & lab[k]).sum() / max(pr.sum() + lab[k].sum(), 1)), 4)}
        arrs[f"{case}_img"], arrs[f"{case}_lab"], arrs[f"{case}_pred"] = img[k].astype(np.uint8), lab[k], pr
        if case in full_volume:
            pv = p.predict_single_npy_array(img[None], {"spacing": [1.0, 1.0, 1.0]}, None, None, False) > 0
            np.savez_compressed(CACHE / f"tk_pred_{case}.npz", pred=pv)
            rec["full_volume_dice_recomputed"] = round(float(2 * (pv & lab).sum() / max(pv.sum() + lab.sum(), 1)), 4)
            rec["full_volume_rel_vol_err_recomputed"] = round(float((pv.sum() - lab.sum()) / lab.sum()), 4)
            zs = (pv ^ lab).sum((1, 2)); kd = int(np.argmax(zs))                                     # worst-disagreement slice
            rec["worst_slice"] = kd
            arrs[f"{case}_img_worst"], arrs[f"{case}_lab_worst"], arrs[f"{case}_pred_worst"] = img[kd].astype(np.uint8), lab[kd], pv[kd]
        out["cases"].append(rec); print(json.dumps(rec), flush=True)
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA / "tk_heldout_slices.npz", **arrs)
    (DATA / "tk_heldout.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tk-infer":
    tk_infer(); sys.exit(0)


# ================================================================ shared: frozen table, box/mask paths, rendering helpers
def load_freeze():
    p = fetch(FREEZE + "brains.jsonl")
    assert sha256(p) == "df1902432f4680b295c1e68cc68804c6d42a71a445d0dbd2003f90bbaf646ead", "freeze hash mismatch"
    rows = {json.loads(l)["key"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}
    assert len(rows) == 689
    return rows


def box_mask_paths(rec):
    """R2 keys of the full-resolution head box and the seg v3 ensemble mask behind one frozen-table row."""
    bk = rec["box_key"]
    if rec["seg_run"].startswith(ALT_RUNS):
        items = {k: b for k, b, _ in json.loads(fetch(ITEMS["alt"]).read_text())}
        return items[bk] + bk + ".npz", "ant-twin/cache/brain_masks_final_alt/" + bk + ".npz"
    items = {k: b for k, b, _ in json.loads(fetch(ITEMS["main"]).read_text()) + json.loads(fetch(ITEMS["v5"]).read_text())}
    return items[bk] + bk + ".npz", "ant-twin/cache/brain_masks_final/" + bk + ".npz"


def load_box_mask(rec):
    import numpy as np
    bkey, mkey = box_mask_paths(rec)
    B, Mk = np.load(fetch(bkey)), np.load(fetch(mkey))
    box, m = B["box"], Mk["mask"] > 0
    assert box.shape == m.shape and list(B["box_zyx"]) == list(Mk["box_zyx"])
    assert int(m.sum()) == rec["brain_vox"], (rec["key"], int(m.sum()), rec["brain_vox"])   # same mask as the frozen volume
    return box, m, float(B["voxel_um"]), [int(v) for v in B["box_zyx"]], bkey, mkey


def nice_bar(width_um, frac=0.28):
    cands = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]
    return min(cands, key=lambda c: abs(c - frac * width_um))


def scalebar(ax, px_um, length_um, color="white", loc="lower right", fs=5.6, pad=0.05, unit_um=True):
    """Scale bar in data coordinates of an imshow axis whose pixels are px_um wide."""
    import matplotlib.patheffects as pe
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    W, H = x1 - x0, y1 - y0; L = length_um / px_um
    xr = x1 - pad * W; xl = xr - L; yb = y1 - pad * H                      # image axes: y increases downwards
    if loc == "lower left":
        xl = x0 + pad * W; xr = xl + L
    ax.plot([xl, xr], [yb, yb], color=color, lw=1.6, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=2.6, foreground="black", alpha=0.6), pe.Normal()])
    lab = (f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm") if unit_um else f"{length_um:g}"
    ax.text((xl + xr) / 2, yb - 0.025 * H, lab, color=color, fontsize=fs, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=1.4, foreground="black", alpha=0.6)])


def window(a, lo=0.5, hi=99.5):
    import numpy as np
    l, h = np.percentile(a, [lo, hi]); return float(l), float(max(h, l + 1))


def square_crop(img, msk, scale=1.35):
    """Square crop centred on the mask bounding box. Side = scale x the larger mask extent, reduced towards the smaller
    box side when the box is narrower (never below 1.05 x the mask extent); area outside the box is NaN (white)."""
    import numpy as np
    yy, xx = np.nonzero(msk)
    cy, cx = (yy.min() + yy.max()) / 2, (xx.min() + xx.max()) / 2
    ext = max(yy.max() - yy.min() + 1, xx.max() - xx.min() + 1)
    side = int(min(scale * ext, max(min(img.shape), 1.05 * ext)))
    y0 = int(round(cy - side / 2)); x0 = int(round(cx - side / 2))
    y0 = min(max(y0, 0), max(img.shape[0] - side, 0)) if side <= img.shape[0] else int(round(cy - side / 2))
    x0 = min(max(x0, 0), max(img.shape[1] - side, 0)) if side <= img.shape[1] else int(round(cx - side / 2))
    out = np.full((side, side), np.nan, np.float32); om = np.zeros((side, side), bool)
    sy0, sx0 = max(y0, 0), max(x0, 0); sy1, sx1 = min(y0 + side, img.shape[0]), min(x0 + side, img.shape[1])
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    om[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = msk[sy0:sy1, sx0:sx1]
    return out, om, (y0, x0, side)


def gallery_selection():
    """Fig. 1c. Pool: the 462 main-analysis workers with a body volume (primary run specimens_table.csv: main_ok and
    body volume present; all in the frozen table). Strata: octiles of log10 body volume. From the smallest octile up, one
    specimen is drawn at random (numpy default_rng(SEED_GALLERY)) among the pool members of that octile whose subfamily
    has not been drawn yet (rows sorted by key), so the 8 specimens come from 8 subfamilies."""
    import numpy as np, pandas as pd
    sp = pd.read_csv(RO / "specimens_table.csv")
    pool = sp[sp.main_ok & sp.body_um3.notna()].sort_values("key").reset_index(drop=True)
    assert len(pool) == 462
    pool["octile"] = pd.qcut(np.log10(pool.body_um3), 8, labels=False)
    rng = np.random.default_rng(SEED_GALLERY); used, picks = set(), []
    for q in range(8):
        cand = pool[(pool.octile == q) & ~pool.subfamily.isin(used)]
        p = cand.iloc[int(rng.integers(len(cand)))]; used.add(p.subfamily)
        picks.append(dict(key=p.key, species=p["name"], subfamily=p.subfamily, voxel_um=float(p.voxel_um), stained=p.stained,
                          body_mm3=float(p.body_um3) / 1e9, body_source="S1" if p.body_source == "S1" else "own v8 mask",
                          brain_mm3=float(p.brain_um3) / 1e9, octile=int(q), n_candidates=int(len(cand))))
    med = float(np.median(np.log10(pool.body_um3)))
    hero = min(picks, key=lambda r: abs(np.log10(r["body_mm3"] * 1e9) - med))["key"]
    return picks, hero, {"pool_n": int(len(pool)), "pool_voxel_um_counts": {str(k): int(v) for k, v in pool.voxel_um.value_counts().items()},
                         "pool_subfamilies": int(pool.subfamily.nunique()), "median_body_mm3": 10 ** med / 1e9}


def render_brain_3d(ax, box, m, vox, lo, hi, elev=22, azim=-55, ds=2, color="#eb6834", contour="#eb6834"):
    """Marching-cubes surface of the mask inside its head box; the three orthogonal box slices through the mask
    centroid are drawn on the back walls (mask outline in colour)."""
    import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mc
    from scipy import ndimage as ndi
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import LightSource
    mm = ndi.gaussian_filter(ndi.zoom(m.astype(np.float32), 1 / ds, order=1), 1.2)
    v, f, _, _ = measure.marching_cubes(mm, 0.5, spacing=(vox * ds,) * 3)
    ax.computed_zorder = False
    Z, Y, X = np.array(box.shape) * vox
    cz, cy, cx = [int(round(float(np.mean(i)))) for i in np.nonzero(m)]
    g = 1                                                   # texture stride on the walls (full resolution)
    crgb = np.array(mc.to_rgba(contour))

    def tex(img, msk):
        a = plt.cm.gray(np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1))
        e = msk & ~ndi.binary_erosion(msk, iterations=2)
        a[e] = crgb
        return a[::g, ::g]
    # plot height = Z - scan z, so that "up" matches the scan views of panel a (slice 0 at the top)
    sl = box[cz]
    ys, xs = np.arange(0, sl.shape[0], g) * vox, np.arange(0, sl.shape[1], g) * vox
    XX, YY = np.meshgrid(xs, ys)
    ax.plot_surface(XX, YY, np.zeros_like(XX), facecolors=tex(sl, m[cz]), rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False, rasterized=True)
    sl = box[:, cy]
    zs, xs = np.arange(0, sl.shape[0], g) * vox, np.arange(0, sl.shape[1], g) * vox
    XX, ZZ = np.meshgrid(xs, zs)
    ax.plot_surface(XX, np.full_like(XX, Y), Z - ZZ, facecolors=tex(sl, m[:, cy]), rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False, rasterized=True)
    sl = box[:, :, cx]
    zs, ys = np.arange(0, sl.shape[0], g) * vox, np.arange(0, sl.shape[1], g) * vox
    YY, ZZ = np.meshgrid(ys, zs)
    ax.plot_surface(np.zeros_like(YY), YY, Z - ZZ, facecolors=tex(sl, m[:, :, cx]), rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False, rasterized=True)
    v = v.copy(); v[:, 0] = Z - v[:, 0]; f = f[:, ::-1]              # mirror in z; keep outward face winding
    tri = v[f][:, :, ::-1]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    sh = LightSource(azdeg=300, altdeg=40).shade_normals(n, fraction=1.0)
    cols = np.clip(np.array(mc.to_rgb(color))[None] * (0.30 + 0.85 * sh[:, None]), 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolor="none", linewidths=0, rasterized=True))
    ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_zlim(0, Z)
    ax.set_box_aspect((X, Y, Z), zoom=1.0)
    ax.set_proj_type("ortho")                               # orthographic, so the scale bar holds along its edge
    ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    L = nice_bar(X, 0.35)                                   # scale bar along the front edge of the floor
    ax.plot([X - L, X], [0, 0], [0, 0], color="black", lw=1.6, solid_capstyle="butt")
    ax.text(X - L / 2, -0.10 * Y, 0, f"{L:g} µm", fontsize=5.6, ha="center", va="top")
    return {"faces": int(len(f)), "slices_zyx": [cz, cy, cx], "scale_bar_um": L}


def outline_text(ax, x, y, s, **kw):
    import matplotlib.patheffects as pe
    return ax.text(x, y, s, path_effects=[pe.withStroke(linewidth=1.2, foreground="black", alpha=0.6)], **kw)


# ================================================================ Fig. 1
def fig1(sel):
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch
    import make_figures as mf
    fr = load_freeze()
    picks, hero = sel["gallery"], sel["hero"]
    hp = next(p for p in picks if p["key"] == hero)
    fig = plt.figure(figsize=(180 * mf.MM, 226 * mf.MM))
    G = fig.add_gridspec(3, 1, height_ratios=[60, 100, 50], hspace=0.20)
    # ---- row 1: (a) low-res whole specimen -> full-res head box; (b) 3D view of the same specimen
    rec = fr[hero]
    box, m, vox, bz, bkey, mkey = load_box_mask(rec)
    lkey = f"ant-twin/cache/lowres/{hero}.npz"
    L = np.load(fetch(lkey)); v = L["img"]
    P, H, W, nlow = int(L["pages"]), int(L["H"]), int(L["W"]), int(L["n_lowres"])
    assert abs(float(L["voxel_um"]) - vox) < 1e-6
    fz, fy, fx = P / nlow, H / v.shape[1], W / v.shape[2]                     # low-res -> full-res (as in finder_predict)
    pz, py, px = vox * fz / 1000, vox * fy / 1000, vox * fx / 1000             # low-res voxel size, mm
    z0, z1, y0, y1, x0, x1 = bz
    g1 = G[0].subgridspec(1, 2, width_ratios=[1.95, 1.0], wspace=0.10)
    zx, zy = v.max(1), v.max(2)                                                # MIP over y (z-x view) and over x (z-y view)
    lo, hi = np.percentile(np.concatenate([zx.ravel(), zy.ravel()]), [2, 99.8])
    wz, wx, wy = v.shape[0] * pz, v.shape[2] * px, v.shape[1] * py
    bw, bh = box.shape[2] * vox, box.shape[1] * vox
    ga = g1[0].subgridspec(1, 3, width_ratios=[wx / wz, wy / wz, 1.25 * bw / bh], wspace=0.10)
    ax1, ax2, ax3 = (fig.add_subplot(ga[i]) for i in range(3))
    mm_ = vox / 1000
    for ax, img, ext, (b0, b1) in ((ax1, zx, wx, (x0 * mm_, x1 * mm_)), (ax2, zy, wy, (y0 * mm_, y1 * mm_))):
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=(0, ext, wz, 0), interpolation="lanczos")
        ax.add_patch(Rectangle((b0, z0 * mm_), b1 - b0, (z1 - z0) * mm_, fill=False, ec=mf.BLUE, lw=1.0))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    ax1.set_title("low-res stack:\nMIP, x–z", fontsize=6.0, pad=2, linespacing=1.1)
    ax2.set_title("\nMIP, y–z", fontsize=6.0, pad=2, linespacing=1.1)
    scalebar(ax1, 1000.0, 1000, loc="lower left")                               # axes in mm: 1 data unit = 1000 um
    kc = (box.shape[0]) // 2
    blo, bhi = window(box)
    ax3.imshow(box[kc], cmap="gray", vmin=blo, vmax=bhi, extent=(0, bw, bh, 0), interpolation="lanczos")
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values(): s.set_edgecolor(mf.BLUE); s.set_linewidth(1.2)
    ax3.set_title(f"full-resolution head box\n({vox:g} µm voxels), middle slice", fontsize=6.0, pad=2, linespacing=1.1)
    scalebar(ax3, 1.0, nice_bar(bw))
    con = ConnectionPatch(xyA=(y1 * mm_, (z0 + z1) / 2 * mm_), coordsA=ax2.transData, xyB=(0, bh / 2), coordsB=ax3.transData,
                          color=mf.BLUE, lw=0.8, arrowstyle="-|>", mutation_scale=7)
    fig.add_artist(con)
    mf.panel(ax1, "a", x=-0.30, y=1.0)
    ax1.text(0.0, -0.015, f"{hp['species']} ({hp['subfamily']}), specimen {hero}", transform=ax1.transAxes, fontsize=5.8,
             style="italic", va="top")
    ax4 = fig.add_subplot(g1[1], projection="3d")
    info3d = render_brain_3d(ax4, box, m, vox, blo, bhi)
    ax4.text2D(0.58, 1.06, "brain mask (seg v3) in the head box, 3D", transform=ax4.transAxes, fontsize=6.0, ha="center", va="bottom")
    ax4.text2D(-0.10, 1.06, "b", transform=ax4.transAxes, fontsize=10, fontweight="bold", va="bottom")
    # ---- rows 2-3: (c) gallery
    gg = G[1].subgridspec(2, 4, hspace=0.30, wspace=0.06)
    tiles = []
    cmap = plt.cm.gray.copy(); cmap.set_bad("white")
    for i, p in enumerate(picks):
        ax = fig.add_subplot(gg[i // 4, i % 4])
        rec = fr[p["key"]]
        box_, m_, vox_, bz_, bk_, mk_ = load_box_mask(rec)
        k = int(np.argmax(m_.sum((1, 2))))                                    # scan slice with the largest brain section
        img, msk, (cy0, cx0, side) = square_crop(box_[k].astype(np.float32), m_[k])
        l_, h_ = window(box_)
        ext = side * vox_
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=l_, vmax=h_, extent=(0, ext, ext, 0), interpolation="lanczos")
        cc = (np.arange(side) + 0.5) * vox_
        ax.contour(cc, cc, msk.astype(float), [0.5], colors=[mf.ORANGE], linewidths=0.8)
        ax.set_xlim(0, ext); ax.set_ylim(ext, 0); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.2 if p["key"] == hero else 0.4); s.set_edgecolor(mf.BLUE if p["key"] == hero else mf.MUTED)
        scalebar(ax, 1.0, nice_bar(ext))
        sp_tex = p["species"].replace(" ", "\\ ")
        ax.set_title(f"$\\it{{{sp_tex}}}$\n{p['subfamily']}, body {p['body_mm3']:.2g} mm³", fontsize=5.9, pad=2.5, linespacing=1.15)
        outline_text(ax, 0.03, 0.97, f"{p['key']}\nvoxel {vox_:g} µm", transform=ax.transAxes, fontsize=4.8, color="white",
                     va="top", linespacing=1.1)
        if i == 0:
            mf.panel(ax, "c", x=-0.10, y=1.12)
        tiles.append(dict(key=p["key"], box_r2=bk_, mask_r2=mk_, slice_z=k, crop_y0_x0_side=[int(cy0), int(cx0), int(side)],
                          brain_vox=int(m_.sum()), brain_mm3_frozen=rec["brain_um3"] / 1e9))
    # ---- row 4: (d, e) held-out accuracy (same data and code as the earlier Fig. 1b-c)
    gd = G[2].subgridspec(1, 2, wspace=0.28)
    mf.draw_accuracy([fig.add_subplot(gd[0]), fig.add_subplot(gd[1])], letters=("d", "e"))
    mf.save(fig, "fig1_scan_to_brain")
    return {"hero": hero, "lowres_r2": lkey, "hero_box_r2": bkey, "hero_mask_r2": mkey, "hero_box_zyx": bz,
            "lowres_shape": list(v.shape), "lowres_to_fullres_factors_zyx": [fz, fy, fx], "render_3d": info3d,
            "gallery_tiles": tiles}


# ================================================================ Fig. S2: seg v3 vs human labels (Toulkeridou et al. 2023)
def figS2():
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import make_figures as mf
    info = json.loads((DATA / "tk_heldout.json").read_text())
    A = np.load(DATA / "tk_heldout_slices.npz")
    cases = info["cases"]
    panels = [(c, "") for c in cases] + [(c, "_worst") for c in cases if c["role"] == "minimum" and "worst_slice" in c]
    fig, axs = plt.subplots(2, 3, figsize=(180 * mf.MM, 138 * mf.MM), squeeze=False, gridspec_kw=dict(wspace=0.05, hspace=0.30))
    for j, (c, suf) in enumerate(panels):
        ax = axs.flat[j]; cs = c["case"]
        mf.panel(ax, "abcde"[j], x=-0.02, y=1.19)
        img, lab, pr = A[f"{cs}_img{suf}"], A[f"{cs}_lab{suf}"], A[f"{cs}_pred{suf}"]
        lo, hi = window(img, 0.5, 99.8)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos")
        ax.contour(lab.astype(float), [0.5], colors=[mf.BLUE], linewidths=0.9)
        ax.contour(pr.astype(float), [0.5], colors=[mf.ORANGE], linewidths=0.8, linestyles=[(0, (3, 1.5))])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4); s.set_edgecolor(mf.MUTED)
        k = c["slice"] if not suf else c["worst_slice"]
        sd = 2 * (lab & pr).sum() / max(lab.sum() + pr.sum(), 1)
        outline_text(ax, 0.03, 0.03, f"slice {k + 1} of {c['n_slices']}, slice Dice {sd:.3f}", transform=ax.transAxes,
                     fontsize=5.6, color="white", va="bottom")
        sp = c["species"].replace("_", " ").replace("crib rinodis", "cribrinodis")
        sp_tex = sp.replace(" ", "\\ ")
        if not suf:
            ax.set_title(f"{c['role']}: {cs} (fold {c['fold']})\n$\\it{{{sp_tex}}}$\n"
                         f"held-out Dice {c['dice']:.3f}, volume error {100 * c['rel_vol_err']:+.1f}%",
                         fontsize=6.0, pad=3, linespacing=1.2)
        else:
            fv, fe = c.get("full_volume_dice_recomputed"), c.get("full_volume_rel_vol_err_recomputed")
            ax.set_title(f"{cs} (minimum): slice with the largest\nlabel–prediction disagreement\n"
                         f"full volume: Dice {fv:.3f}, volume error {100 * fe:+.1f}%",
                         fontsize=6.0, pad=3, linespacing=1.2)
    ax = axs.flat[5]; ax.set_axis_off()
    ax.legend(handles=[Line2D([], [], color=mf.BLUE, lw=1.2, label="human label\n(Toulkeridou et al. 2023)"),
                       Line2D([], [], color=mf.ORANGE, lw=1.2, ls=(0, (3, 1.5)), label="seg v3, model of the\nheld-out fold")],
              loc="upper left", fontsize=6.4, frameon=False, labelspacing=1.2)
    chk = [c for c in cases if "full_volume_dice_recomputed" in c]
    ax.text(0.02, 0.45, "Re-predicted in full on CPU from the\nstored checkpoints (stored → recomputed):\n" +
            "\n".join(f"{c['case']}: Dice {c['dice']:.4f} → {c['full_volume_dice_recomputed']:.4f}" for c in chk) +
            "\n\nImages: cropped and resampled to\n520 × 520 px by the data set authors;\nno physical scale available.",
            transform=ax.transAxes, fontsize=5.8, va="top", color=mf.INK2, linespacing=1.25)
    mf.save(fig, "figS2_tk_heldout")
    return {"cases": [{k: c[k] for k in c if k not in ("image_r2", "label_r2", "checkpoint_r2")} | {
        "image_r2": c["image_r2"], "label_r2": c["label_r2"], "checkpoint_r2": c["checkpoint_r2"]} for c in cases],
        "stats": info["stats_117_tk_cases"]}


def species_names():
    """key -> species name for all 820 colony workers (specimen table v2, runs/20261113-153618_*/outputs/specimens.csv)."""
    import pandas as pd
    d = pd.read_csv(fetch("ant-twin/runs/20261113-153618_specimen-table-v2-fix-species-name-joins_DESKTOP-7Q7S0N3/outputs/specimens.csv"))

    def vial_code(v):                                                         # as in stages/lowres_cache.py
        p = str(v).split("-")
        return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)
    return {f"{r.folder}__{vial_code(r.vial)}": r.name for r in d.itertuples()}


# ================================================================ Fig. S3: cuticle QC rule (owners' program)
CUT_RUN_DIRS = None


def cuticle_records():
    """Adaptive and strict rules exactly as in scripts/robustness_v1/r3x_run.py, recomputed from cuticle_final.jsonl."""
    import numpy as np, pandas as pd
    rec = pd.DataFrame([json.loads(l) for l in fetch(CUT_FINAL).read_text().splitlines() if l.strip()])
    rec = rec[rec.status == "ok"].drop_duplicates("key", keep="last").set_index("key")
    within = lambda a, b: (a > 0) & (b > 0) & ((a / b.where(b > 0)).between(1 / 1.10, 1.10))
    use110 = within(rec.t110_body_um3, rec.t130_body_um3); use145 = ~use110 & within(rec.t145_body_um3, rec.t130_body_um3)
    rec["thr_used"] = np.where(use110, "t110", np.where(use145, "t145", "none"))
    for q in ("body", "cuticle"):
        rec[f"ad_{q}_um3"] = np.where(use110, rec[f"t110_{q}_um3"], np.where(use145, rec[f"t145_{q}_um3"], np.nan))
    rec["qc_pass"] = (rec.thr_used != "none") & (rec.ad_cuticle_um3 / rec.ad_body_um3).between(0.03, 0.40)
    rec["r110"] = rec.t110_body_um3 / rec.t130_body_um3; rec["r145"] = rec.t145_body_um3 / rec.t130_body_um3
    assert len(rec) == 185 and int(rec.qc_pass.sum()) == 141
    assert rec[rec.qc_pass].thr_used.value_counts().to_dict() == {"t110": 92, "t145": 49}
    return rec


def cuticle_run_dir(run):
    global CUT_RUN_DIRS
    if CUT_RUN_DIRS is None:
        import r2
        p = r2c().get_paginator("list_objects_v2"); CUT_RUN_DIRS = {}
        for page in p.paginate(Bucket=r2.BUCKET, Prefix="ant-twin/runs/20261115-", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                CUT_RUN_DIRS[cp["Prefix"].split("/")[-2]] = cp["Prefix"]
    return CUT_RUN_DIRS[run]


def cuticle_selection():
    """Accepted: drawn at random (default_rng(SEED_CUTICLE), keys sorted) among specimens that passed the adaptive rule
    with threshold 110 (the threshold shown on the QC sheets). Rejected: drawn among specimens excluded because neither
    threshold gave a stable body volume and body(110)/body(130) > 1.10 (bright medium entering the mask at 110)."""
    import numpy as np
    rec = cuticle_records()
    acc = sorted(rec.index[rec.qc_pass & (rec.thr_used == "t110")])
    rej = sorted(rec.index[(rec.thr_used == "none") & (rec.r110 > 1.10)])
    rng = np.random.default_rng(SEED_CUTICLE)
    a = acc[int(rng.integers(len(acc)))]; r = rej[int(rng.integers(len(rej)))]
    return {"accepted": a, "rejected": r, "n_accepted_t110": len(acc), "n_rejected_leak": len(rej),
            "n_none": int((rec.thr_used == "none").sum()), "n_fail_fraction": int(((rec.thr_used != "none") & ~rec.qc_pass).sum())}


def figS3(cs):
    import numpy as np, matplotlib.pyplot as plt
    from PIL import Image
    import make_figures as mf
    rec = cuticle_records()
    tp = json.loads(fetch("ant-twin/meta/true_pages.json").read_text())
    names = species_names()
    fig = plt.figure(figsize=(180 * mf.MM, 150 * mf.MM))
    G = fig.add_gridspec(2, 1, hspace=0.34)
    out = {}
    cols = {"t110": mf.ORANGE, "t130": mf.MUTED, "t145": mf.BLUE}
    for row, (role, key) in enumerate((("accepted", cs["accepted"]), ("rejected", cs["rejected"]))):
        r = rec.loc[key]; run = r["run"]; rd = cuticle_run_dir(run)
        qk, sk = rd + f"outputs/qc/{key}.jpg", rd + f"outputs/slices/{key}.npz"
        im = np.asarray(Image.open(fetch(qk)).convert("RGB")); S = np.load(fetch(sk))
        Hq, Wq = im.shape[0] // 4, im.shape[1] // 2
        W_full = int(tp[key]["W"]); px_um = float(r["voxel_um"]) * W_full / Wq      # one QC-tile pixel in um
        g0 = G[row].subgridspec(1, 2, width_ratios=[4.0, 1.55], wspace=0.16)
        g = g0[0].subgridspec(2, 4, wspace=0.04, hspace=0.04)
        gap = int(S["gap"])
        lo = min(int(r[f"{t}_beginning"]) for t in ("t110", "t130", "t145")); hi = max(int(r[f"{t}_end"]) for t in ("t110", "t130", "t145"))
        idx = list(range(lo, hi, gap)); qsel = sorted({idx[j] for j in np.linspace(0, len(idx) - 1, 4).astype(int)})
        for j in range(4):
            for i, half in enumerate((0, 1)):                                    # 0: raw slice, 1: segmentation overlay
                ax = fig.add_subplot(g[i, j])
                ax.imshow(im[j * Hq:(j + 1) * Hq, half * Wq:(half + 1) * Wq], interpolation="lanczos")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_linewidth(0.4); s.set_edgecolor(mf.MUTED)
                if i == 0:
                    ax.set_title(f"slice {qsel[j] + 1}", fontsize=5.6, pad=1.5)
                if j == 0:
                    ax.set_ylabel(("scan slice", "threshold 110:\ncuticle green,\nsoft tissue blue")[i], fontsize=5.6)
                if i == 1 and j == 3:
                    scalebar(ax, px_um, 1000 if W_full * float(r["voxel_um"]) > 4000 else 500, pad=0.09)
        ax = fig.add_subplot(g0[1])
        for t in ("t110", "t130", "t145"):
            b = S[f"{t}_body"].astype(float); z0 = int(S[f"{t}_beginning"])
            zz = (z0 + gap * np.arange(len(b))) * float(r["voxel_um"]) / 1000
            ax.plot(zz, b * float(r["voxel_um"]) ** 2 / 1e6, color=cols[t], lw=0.9,
                    label=f"{t[1:]}: body {r[f'{t}_body_um3'] / 1e9:.3g} mm³")
        for q in qsel:
            ax.axvline(q * float(r["voxel_um"]) / 1000, color=mf.GRID, lw=0.6, zorder=0)
        ax.set_xlabel("position along the scan axis (mm)", fontsize=6.0); ax.set_ylabel("body cross-section per slice (mm²)", fontsize=6.0)
        ax.set_ylim(0, 1.55 * ax.get_ylim()[1])
        ax.legend(fontsize=5.2, loc="upper left", title="threshold: body volume", title_fontsize=5.2, ncol=1, frameon=False,
                  alignment="left")
        ax.tick_params(labelsize=5.6)
        verdict = (f"Accepted by the adaptive rule at threshold 110: body(110)/body(130) = {r['r110']:.2f}, "
                   f"cuticle/body = {r['ad_cuticle_um3'] / r['ad_body_um3']:.2f}."
                   if role == "accepted" else
                   f"Rejected: body(110)/body(130) = {r['r110']:.2f} and body(145)/body(130) = {r['r145']:.2f}, both outside 1/1.10–1.10.")
        sp = names.get(key, "")
        yh = G[row].get_position(fig).y1 + 0.03
        fig.text(0.02, yh - 0.004, "ab"[row], fontsize=10, fontweight="bold", ha="left", va="bottom")
        fig.text(0.06, yh, f"$\\it{{{sp.replace(' ', '~')}}}$, {key}, {r['voxel_um']:g} µm voxels. {verdict}",
                 fontsize=6.3, ha="left", va="bottom")
        out[role] = {"key": key, "run": run, "qc_r2": qk, "slices_r2": sk, "thr_used": r["thr_used"], "r110": float(r["r110"]),
                     "r145": float(r["r145"]), "qc_slices": qsel, "voxel_um": float(r["voxel_um"]),
                     "body_um3": {t: float(r[f"{t}_body_um3"]) for t in ("t110", "t130", "t145")}}
    mf.save(fig, "figS3_cuticle_qc")
    return out


# ================================================================ Fig. S4: masks rejected by the vision-agent QC
V5_RUNS = ("ant-twin/runs/20261115-065118_seg-final-v5-ensemble-on-boxes-for-finder-v8-det_5d5eb7649d55/",
           "ant-twin/runs/20261115-065120_seg-final-v5-reverse-second-gpu_668c59782304/")


def qc_selection():
    """Final QC round on seg v3 ensemble masks (finder v8 boxes, meta/brains_v5_auto_review/, 268 masks). Rejected =
    not (mask good and box on head and brain fully in box); restricted to specimens absent from the frozen table and to
    non-empty masks. Three categories by the agent's box verdict (on head: no / unsure / yes); one mask per category is
    drawn at random (default_rng(SEED_QC), keys sorted)."""
    import numpy as np
    fr = load_freeze(); in_table = set(fr) | {r["box_key"] for r in fr.values()}
    v = []
    for i in range(1, 6):
        v += json.loads(fetch(f"ant-twin/meta/brains_v5_auto_review/auto_review_{i}.json").read_text())
    assert len(v) == 268
    vox = {json.loads(l)["key"]: json.loads(l) for l in fetch("ant-twin/review/brains_v5/brains.jsonl").read_text().splitlines() if l.strip()}
    acc = lambda r: r["mask"] == "good" and r["box_on_head"] is True and r["brain_fully_in_box"] is True
    rej = [r for r in v if not acc(r)]
    cats = {"box not on the head": lambda r: r["box_on_head"] is False,
            "box position unsure": lambda r: r["box_on_head"] == "unsure",
            "box on the head, mask rejected": lambda r: r["box_on_head"] is True}
    rng = np.random.default_rng(SEED_QC); picks = []; counts = {}
    for name, f in cats.items():
        allc = [r for r in rej if f(r)]
        pool = sorted((r for r in allc if r["key"] not in in_table and vox[r["key"]]["brain_vox"] > 0), key=lambda r: r["key"])
        counts[name] = {"rejected": len(allc), "eligible": len(pool)}
        picks.append({"category": name, **pool[int(rng.integers(len(pool)))]})
    return picks, {"n_reviewed": len(v), "n_accepted": sum(acc(r) for r in v), "n_rejected": len(rej), "categories": counts}


def figS4(qs):
    import numpy as np, matplotlib.pyplot as plt, textwrap
    from PIL import Image
    import make_figures as mf
    picks, stats = qs
    names = species_names()
    fig = plt.figure(figsize=(180 * mf.MM, 180 * mf.MM))
    G = fig.add_gridspec(len(picks), 2, width_ratios=[2.25, 1.0], hspace=0.16, wspace=0.03)
    out = []
    for i, p in enumerate(picks):
        key = p["key"]; rk = f"ant-twin/review/brains_v5/qc/{key}.jpg"
        src = fetch(rk); h = sha256(src); run_copy = None
        for run in V5_RUNS:                                                  # the same sheet as written by the seg run
            try:
                if sha256(fetch(run + f"outputs/qc/{key}.jpg")) == h:
                    run_copy = run + f"outputs/qc/{key}.jpg"; break
            except Exception:
                pass
        im = np.asarray(Image.open(src).convert("RGB"))
        ax = fig.add_subplot(G[i, 0]); ax.imshow(im, interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4); s.set_edgecolor(mf.MUTED)
        bb = G[i, 0].get_position(fig)
        fig.text(0.02, bb.y1, "abc"[i], fontsize=10, fontweight="bold", va="top", ha="left")
        tx = fig.add_subplot(G[i, 1]); tx.set_axis_off()
        yn = lambda x: {True: "yes", False: "no"}.get(x, str(x))
        ev = p.get("est_vol_error_pct")
        sp = names.get(key, "")
        sp_tex = sp.replace(" ", "~")
        body = (f"specimen {key}\n$\\it{{{sp_tex}}}$\n\n"
                f"agent verdict (verbatim fields):\n"
                f"  mask: {p['mask']}\n  box on head: {yn(p['box_on_head'])}\n  brain fully in box: {yn(p['brain_fully_in_box'])}\n"
                f"  estimated volume error: {'not given' if ev is None else f'{ev:+d}%'}\n  confidence: {p['confidence']:.2f}\n\n"
                "agent note:\n" + "\n".join(textwrap.wrap(f"“{p['note']}”", 36)))
        tx.text(0.04, 1.0, p["category"], fontsize=6.6, fontweight="bold", va="top", ha="left", transform=tx.transAxes)
        tx.text(0.04, 0.88, body, fontsize=6.0, va="top", ha="left", linespacing=1.25, transform=tx.transAxes)
        out.append({"key": key, "category": p["category"], "qc_sheet_r2": rk, "qc_sheet_run_copy_r2": run_copy, "sha256": h,
                    "verdict": {k: p[k] for k in ("mask", "box_on_head", "brain_fully_in_box", "est_vol_error_pct", "confidence", "note")}})
    mf.save(fig, "figS4_qc_rejections")
    return {"picks": out, "stats": stats}


# ================================================================ driver
FIG_FILES = ["fig1_scan_to_brain", "figS1_pipeline_schematic", "figS2_tk_heldout", "figS3_cuticle_qc", "figS4_qc_rejections"]


def upload():
    """Figure files (PNG 300 dpi + PDF) and their provenance records -> R2 ant-twin/preprint/v1/figures/ (nothing else)."""
    import r2
    ctype = {".png": "image/png", ".pdf": "application/pdf", ".json": "application/json", ".npz": "application/octet-stream"}
    files = [OUT / f"{n}.{ext}" for n in FIG_FILES for ext in ("png", "pdf")]
    files += [DATA / "image_figures_selection.json", DATA / "tk_heldout.json", DATA / "tk_heldout_slices.npz"]
    for p in files:
        key = UPLOAD_PREFIX + ("image_data/" if p.parent == DATA else "") + p.name
        r2c().upload_file(str(p), r2.BUCKET, key, ExtraArgs={"ContentType": ctype[p.suffix]})
        print("uploaded", key, p.stat().st_size)


def main():
    DATA.mkdir(exist_ok=True)
    picks, hero, gstats = gallery_selection()
    sel = {"gallery": picks, "hero": hero}
    record = {"note": "Inputs and selections of the image figures (make_image_figures.py). R2 bucket dsc-biology.",
              "frozen_table_r2": FREEZE + "brains.jsonl",
              "frozen_table_sha256": "df1902432f4680b295c1e68cc68804c6d42a71a445d0dbd2003f90bbaf646ead",
              "fig1": {"selection_rule": gallery_selection.__doc__.strip(), "seed": SEED_GALLERY, "pool": gstats, "picks": picks,
                       "hero_rule": "panels a-b: the gallery specimen closest to the pool median of log10 body volume"}}
    record["fig1"].update(fig1(sel))
    record["figS2"] = {"selection_rule": tk_select.__doc__.strip(), "inference": tk_infer.__doc__.strip(), **figS2()}
    cs = cuticle_selection()
    record["figS3"] = {"selection_rule": cuticle_selection.__doc__.strip(), "seed": SEED_CUTICLE, **cs, **figS3(cs)}
    qs = qc_selection()
    record["figS4"] = {"selection_rule": qc_selection.__doc__.strip(), "seed": SEED_QC, **figS4(qs)}
    (DATA / "image_figures_selection.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print("wrote", DATA / "image_figures_selection.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    if cmd == "render":
        main()
    elif cmd == "upload":
        upload()
    elif cmd == "select":
        print(json.dumps({"gallery": gallery_selection(), "cuticle": cuticle_selection(), "qc": qc_selection()}, indent=1, default=str))
