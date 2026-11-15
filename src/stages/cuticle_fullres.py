"""Stage cuticle_fullres: whole-body, cuticle and soft-tissue volume of a worker at full resolution.

Why: robustness check R3 (brain relative to soft tissue) needs body minus cuticle for every species; the owners' S1
gives it only for part of them. Validation target: owners' S1 body_volume and cuticle_volume on the overlap.
Per specimen:
1. bbox of the body mask of the body_volume v8 segmentation on the low-res cache (+ margin), mapped to
   full-res rows/cols with the true page count;
2. full-res slices: uncompressed BigTIFF -> HTTP-range rows of the bbox, every zstride-th page; compressed -> whole-file
   download, every zstride-th page;
3. per slice on GPU (cupy; CPU fallback, same ops; radii in um): Gaussian sigma_px, multi-Otsu (t0 background|tissue,
   t1 tissue|cuticle, on a subsample of the stack), tube = fill(g > t0) minus wall_um (Euclidean distance), shell = (g > t1) & tube, closing
   close_um, body = fill(shell) (leak guard: if the fill exceeds 0.6 x tube area keep the shell only), cuticle = shell
   before closing inside body; 3D: keep components >= 3% of the largest;
4. volumes = voxel counts x vox^3 x z step (z step recorded per specimen).
Config: specimens_run, keys_key (R2 list of keys, optional), true_pages_key, lowres_prefix, zstride (minimum z step),
target_gb (z step grows until the uint8 stack <= target_gb), zstride_max, sigma_px, wall_um, close_um, margin_um, workers,
fetch_threads, gpu, shard, qc_every.
Outputs: cuticle.jsonl, qc/<key>.jpg, summary.json.
"""
import io, json, time, tempfile, os, shutil, pathlib, urllib.request, concurrent.futures as cf
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import r2
import bigtiff_range as BT

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def lowres_bbox(v):
    """bbox of the body mask from the validated body_volume v8 low-res segmentation (S1 median ratio 1.04)."""
    from stages.body_volume import segment
    body, _ = segment(v)
    if not body.any(): return None
    zz, yy, xx = np.nonzero(body)
    return [zz.min(), zz.max() + 1, yy.min(), yy.max() + 1, xx.min(), xx.max() + 1]


def _backend(cfg):
    """GPU (cupy) when available and cfg gpu != false, else CPU; same operations on both."""
    if cfg.get("gpu", True):
        try:
            os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")   # Blackwell (sm_120): JIT from PTX
            import cupy as cp
            from cupyx.scipy import ndimage as cnd
            cp.zeros(1).sum(); return cp, cnd, True
        except Exception:
            pass
    return np, ndi, False


def _fill(xp, nd, m):
    """fill holes = everything not connected to the image border through background (4-connectivity)."""
    lab, n = nd.label(~m)
    if int(n) == 0: return m.copy()
    edge = xp.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    lut = xp.zeros(int(n) + 1, bool); lut[edge] = True; lut[0] = False
    return ~lut[lab]


def _disk(xp, r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; return xp.asarray(xx * xx + yy * yy <= r * r)


def segment_slices(stack, vox, cfg):
    from skimage.filters import threshold_multiotsu
    xp, nd, gpu = _backend(cfg)
    sig = float(cfg.get("sigma_px", 1.0))
    sub = stack[:: max(1, len(stack) // 40), ::2, ::2].astype(np.float32)
    t0, t1 = (float(x) for x in threshold_multiotsu(ndi.gaussian_filter(sub, sig), classes=3))
    wall = cfg.get("wall_um", 70) / vox; close = _disk(xp, max(1, int(round(cfg.get("close_um", 30) / vox))))
    body = np.zeros(stack.shape, bool); cut = np.zeros(stack.shape, bool); guarded = 0
    for z in range(len(stack)):
        g = nd.gaussian_filter(xp.asarray(stack[z], dtype=xp.float32), sig)
        tube = _fill(xp, nd, g > t0)
        inner = (nd.distance_transform_edt(tube) > wall) if bool(tube.any()) else tube   # tube interior minus wall_um
        sh = (g > t1) & inner
        f = _fill(xp, nd, nd.binary_closing(sh, structure=close))
        if float(f.sum()) > 0.6 * max(float(inner.sum()), 1.0): f = sh; guarded += 1   # fill leaked into the tube
        c = sh & f
        body[z] = f.get() if gpu else f; cut[z] = c.get() if gpu else c
    # 3D components on a 2x2 in-plane max-pooled mask (keeps connectivity, 1/4 of the label memory)
    Z, H, W = body.shape; pb = np.zeros((Z, (H + 1) // 2, (W + 1) // 2), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            q = body[:, dy::2, dx::2]; pb[:, : q.shape[1], : q.shape[2]] |= q
    lab, n = ndi.label(pb)
    if n:
        sizes = ndi.sum(pb, lab, range(1, n + 1)); kp = np.isin(lab, [i + 1 for i in range(n) if sizes[i] >= 0.03 * sizes.max()])
        for z in range(Z):
            k = np.repeat(np.repeat(kp[z], 2, 0), 2, 1)[:H, :W]; body[z] &= k; cut[z] &= k
        del kp
    del lab, pb
    return body, cut, {"t0": round(t0, 2), "t1": round(t1, 2), "guarded_frac": round(guarded / max(len(stack), 1), 4), "n_comp": int(n), "gpu": gpu}


def crop_parallel(url, info, z0, z1, y0, y1, x0, x1, zstep, threads=16):
    """BT.crop with concurrent HTTP range requests (one per page)."""
    W = info["W"]; zs = list(range(z0, z1, zstep))
    def page(z):
        st = BT.data_offset(info, z) + y0 * W
        return np.frombuffer(BT.get(url, st, st + (y1 - y0) * W - 1), np.uint8).reshape(y1 - y0, W)[:, x0:x1].copy()
    with cf.ThreadPoolExecutor(threads) as ex: return np.stack(list(ex.map(page, zs)))


def qc_jpg(stack, body, cut, path):
    ks = np.nonzero(body.any((1, 2)))[0]
    if not len(ks): return
    tiles = []
    for k in np.linspace(ks.min(), ks.max(), 4).astype(int):
        a = stack[k].astype(np.float32); a = (a - a.min()) / max(np.ptp(a), 1); rgb = np.stack([a] * 3, -1)
        e = body[k] & ~ndi.binary_erosion(body[k]); rgb[e] = [1, 0.1, 0.1]; rgb[cut[k]] = rgb[cut[k]] * 0.5 + np.array([0, 0.5, 0])
        tiles.append((rgb * 255).astype(np.uint8))
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    canvas = np.zeros((h, w * 4, 3), np.uint8)
    for i, t in enumerate(tiles): canvas[: t.shape[0], i * w: i * w + t.shape[1]] = t
    im = Image.fromarray(canvas); im.thumbnail((2000, 600)); im.save(path, quality=85)


def main(cfg, out):
    import pandas as pd
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    d = pd.read_csv(io.BytesIO(get(f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")))
    d = d[(d.caste == "worker") & d.url_full.notna()].copy()
    d["key"] = [f"{f}__{vial_code(v)}" for f, v in zip(d.folder, d.vial)]
    if cfg.get("keys_key"):
        want = set(json.loads(get(cfg["keys_key"]))); d = d[d.key.isin(want)]
    recs = d.drop_duplicates("key").sort_values("key").to_dict("records")
    si, sn = cfg.get("shard", [0, 1]); recs = recs[si::sn]
    tp = json.loads(get(cfg["true_pages_key"])); lp = cfg.get("lowres_prefix", "ant-twin/cache/lowres/")
    (out / "qc").mkdir(exist_ok=True); zs_min = int(cfg.get("zstride", 2))
    print(f"specimens {len(recs)} (shard {si}/{sn})", flush=True)

    def one(i_r):
        i, r = i_r; key = r["key"]; t0 = time.time(); vox = float(r["voxel_um"])
        base = {"key": key, "specimen_code": r.get("specimen_code"), "voxel_um": vox}
        try:
            if key not in tp: return {**base, "status": "no_true_pages"}
            P, H, W = int(tp[key]["pages"]), int(tp[key]["H"]), int(tp[key]["W"])
            z = np.load(io.BytesIO(get(lp + key + ".npz"))); v = z["img"]
            bb = lowres_bbox(v)
            if bb is None: return {**base, "status": "no_specimen_lowres"}
            f = np.array([P / v.shape[0], H / v.shape[1], W / v.shape[2]]); m = cfg.get("margin_um", 150) / vox
            z0, z1 = max(0, int(bb[0] * f[0] - m)), min(P, int(np.ceil(bb[1] * f[0] + m)))
            y0, y1 = max(0, int(bb[2] * f[1] - m)), min(H, int(np.ceil(bb[3] * f[1] + m)))
            x0, x1 = max(0, int(bb[4] * f[2] - m)), min(W, int(np.ceil(bb[5] * f[2] + m)))
            gb1 = (z1 - z0) * (y1 - y0) * (x1 - x0) / 1e9                       # uint8 GB of the box at every page
            zs = max(zs_min, int(np.ceil(gb1 / cfg.get("target_gb", 4))))          # adaptive z step: stack <= target_gb
            if zs > cfg.get("zstride_max", 8): return {**base, "status": "too_big", "gb_full": round(gb1, 2)}
            url = r["url_full"]; info = BT.layout(url)
            if info["compression"] == 1:
                if not BT.verify_offset(url, info, z1 - 1): return {**base, "status": "layout_irregular"}
                stack = crop_parallel(url, info, z0, z1, y0, y1, x0, x1, zs, cfg.get("fetch_threads", 16)); how = "range"
            else:
                import tifffile
                tmp = pathlib.Path(tempfile.gettempdir()) / f"cut_{key}.tif"
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1800) as resp, open(tmp, "wb") as fh: shutil.copyfileobj(resp, fh, 1 << 22)
                    with tifffile.TiffFile(tmp) as tf:
                        if len(tf.pages) != P: return {**base, "status": "page_mismatch", "pages_file": len(tf.pages), "pages_true": P}
                        stack = np.stack([tf.pages[k].asarray()[y0:y1, x0:x1] for k in range(z0, z1, zs)])
                finally:
                    if tmp.exists(): os.remove(tmp)
                how = "full_file"
            body, cut, info_s = segment_slices(stack, vox, cfg)
            v3 = vox ** 3 * zs
            rec = {**base, "status": "ok", "fetch": how, "zstride": zs, "box_zyx": [z0, z1, y0, y1, x0, x1], "stack_shape": list(stack.shape),
                   "body_um3": float(body.sum() * v3), "cuticle_um3": float(cut.sum() * v3), "soft_um3": float((body & ~cut).sum() * v3),
                   **info_s, "sec": round(time.time() - t0, 1)}
            if i % cfg.get("qc_every", 1) == 0: qc_jpg(stack, body, cut, out / "qc" / f"{key}.jpg")
            return rec
        except Exception as ex:
            return {**base, "status": "error", "error": repr(ex)[:200]}

    with cf.ThreadPoolExecutor(cfg.get("workers", 4)) as ex, open(out / "cuticle.jsonl", "w", encoding="utf-8") as fo:
        for rec in ex.map(one, enumerate(recs)):
            fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec)[:300], flush=True)
    rows = [json.loads(l) for l in open(out / "cuticle.jsonl", encoding="utf-8")]
    from collections import Counter
    summ = {"n": len(rows), "status": dict(Counter(r["status"] for r in rows))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
