"""Stage tiny_ends: finder-free candidate brain boxes for tiny workers the low-res finder misses.

For each target specimen: specimen mask from the cached low-res stack (cuticle class, tube wall removed, as in
body_volume v6), long axis by PCA of the mask voxels (physical units), and for each end the axis-aligned bbox of the
voxels whose projection lies in the outer `end_frac` of the length. Each end becomes a full-res box candidate written
as a finder.jsonl-compatible row with key "<key>@A" / "<key>@B"; boxes_v3 and seg_predict then process them, and the
end with a compact brain is chosen after agent QC.
Config: tags (two end suffixes, default A/B), targets_key + finder_runs (explicit key list; rows of the finder runs supply the specimen fields), true_pages_key
        (exact page count), or (older) specimens_run, targets_from_runs (finder runs whose status==no_brain keys are targets, minus keys rescued ok in
        rescue_runs), rescue_runs, cache_prefix, end_frac, margin_frac.
Outputs: finder.jsonl (two rows per specimen), summary.json.
"""
import io, json
import numpy as np
from scipy import ndimage as ndi
import r2
from stages.finder_fast import vial_code


def specimen_mask(v):
    from skimage.filters import threshold_multiotsu
    from skimage.morphology import disk
    g = ndi.gaussian_filter(v.astype(np.float32), 1.0)
    t0, t1 = (float(x) for x in threshold_multiotsu(g[::2, ::4, ::4], classes=3))
    m = np.zeros(g.shape, bool)
    for z in range(g.shape[0]):
        tube = ndi.binary_fill_holes(g[z] > t0); inner = ndi.binary_erosion(tube, structure=disk(1), iterations=6)
        sh = ndi.binary_closing((g[z] > t1) & inner, structure=disk(3)); f = ndi.binary_fill_holes(sh)
        m[z] = sh if f.sum() > 0.6 * max(inner.sum(), 1) else f
    lab, n = ndi.label(m)
    if n == 0: return m
    s = ndi.sum(m, lab, range(1, n + 1)); return np.isin(lab, [i + 1 for i in range(n) if s[i] >= 0.03 * s.max()])


def main(cfg, out):
    import pandas as pd
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    rows = lambda run: [json.loads(l) for l in get(f"ant-twin/runs/{run}/outputs/finder.jsonl").decode().splitlines() if l.strip()]
    if cfg.get("targets_key"):  # explicit key list; finder rows (any status) of finder_runs give the specimen fields
        want = set(json.loads(get(cfg["targets_key"]))); targets = {}
        for run in cfg.get("finder_runs", []):
            for r in rows(run):
                if r["key"] in want: targets[r["key"]] = r
    else:
        targets = {r["key"]: r for run in cfg["targets_from_runs"] for r in rows(run) if r.get("status") == "no_brain"}
        for run in cfg.get("rescue_runs", []):
            for r in rows(run):
                if r.get("status") == "ok": targets.pop(r["key"], None)
    TP = json.loads(get(cfg["true_pages_key"])) if cfg.get("true_pages_key") else {}
    print(f"targets: {len(targets)}", flush=True)
    ef, mf = cfg.get("end_frac", 0.35), cfg.get("margin_frac", 0.15)
    with open(out / "finder.jsonl", "w", encoding="utf-8") as fo:
        for key, r in sorted(targets.items()):
            try:
                z = np.load(io.BytesIO(get(cfg.get("cache_prefix", "ant-twin/cache/lowres/") + key + ".npz")))
            except Exception:
                print("not cached", key); continue
            v, nlow, P, H, W = z["img"], int(z["n_lowres"]), int(z["pages"]), int(z["H"]), int(z["W"])
            if TP:
                if key not in TP: print("no true pages", key); continue
                P = int(TP[key]["pages"])  # cached page estimate is wrong for compressed stacks
            f = np.array([P / nlow, H / v.shape[1], W / v.shape[2]]); sp = float(r["voxel_um"]) * f
            m = specimen_mask(v); pts = np.argwhere(m)
            if len(pts) < 100: fo.write(json.dumps({**r, "key": key + "@A", "status": "no_specimen"}) + "\n"); continue
            ph = pts * sp; ctr = ph.mean(0); u = np.linalg.svd(ph - ctr, full_matrices=False)[2][0]  # long axis (um)
            proj = (ph - ctr) @ u; lo_p, hi_p = proj.min(), proj.max(); L = hi_p - lo_p
            ta, tb = cfg.get("tags", ["A", "B"])
            for tag, sel in ((ta, proj <= lo_p + ef * L), (tb, proj >= hi_p - ef * L)):
                q = pts[sel]; lo = q.min(0).astype(float); hi = q.max(0).astype(float) + 1
                half = (hi - lo) / 2 * (1 + mf); cc = (lo + hi) / 2
                c_ = cc * f; h_ = half * f; lim = np.array([P, H, W])
                z0, y0, x0 = np.maximum(np.floor(c_ - h_), 0).astype(int); z1, y1, x1 = np.minimum(np.ceil(c_ + h_), lim).astype(int)
                rec = {**{k: r[k] for k in ("species", "species_key", "subfamily", "stained", "voxel_um", "folder", "vial", "url_full") if k in r},
                       "key": f"{key}@{tag}", "parent_key": key, "status": "ok", "box_zyx": [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                       "length_um": round(float(L), 1), "source": "tiny_ends"}
                fo.write(json.dumps(rec) + "\n")
            print(key, "length um", round(float(L)), flush=True)
    n = sum(1 for _ in open(out / "finder.jsonl", encoding="utf-8"))
    (out / "summary.json").write_text(json.dumps({"targets": len(targets), "rows": n}, indent=1)); print("rows", n, flush=True)
