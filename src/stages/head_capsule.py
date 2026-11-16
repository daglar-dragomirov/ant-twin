"""Stage head_capsule: head capsule volume (cuticle + contents, without appendages) on the low-res AntScan stacks.

Seed: the approved full-res brain mask (inside the head by construction). Per specimen:
1. ROI = brain centre +- k_roi x the largest brain half-extent, cut from the low-res stack (true page count for z),
   resampled to isotropic voxels (side <= max_side); the ROI grows x1.5 (twice at most) while the head touches an
   inner ROI face.
2. Body mask: cuticle = smoothed intensity > threshold, closed; every 2D slice along each of the three axes is
   hole-filled and the fills are united (a hole fill never leaks outward); 3D hole fill; the component holding the
   brain. Thresholds tried from high to low (multi-Otsu 3 classes: t_hi, (t_lo + t_hi)/2, t_lo + (t_hi - t_lo)/4),
   closing 1 then 2 voxels; the first mask holding the brain and filling < 60 % of the ROI is used.
3. Head = the brain-holding part of that mask, split from the thorax and appendages by a marker watershed on the
   negative distance map D (split lines run along the narrowest constrictions: neck, appendage joints). Dc = 90th
   percentile of D over the brain. Three variants (the vision agent picks the best per specimen):
     A "core":  markers = component of {D >= 0.5 Dc} holding the brain (head) vs. all other components of it and the
                ROI border (rest); then an opening of radius 0.15 Dc removes thin attachments (antennae).
     B "hmax":  markers = h-maxima of D (h = 0.08 Dc); head = basins whose maximum lies in the head core; opening 0.15 Dc.
     C "interior": the same on the interior (body minus a 1-voxel-dilated cuticle), so parts that only touch the head
                stay apart by their two cuticle walls; the head's own wall (3 voxels) is added back, closing 0.15 Dc,
                3D hole fill.
   Volume = voxel count x voxel volume (isotropic ROI spacing).
Config: items_key (R2 JSON [[key, brain_mask_npz_key], ...]), true_pages_key, cache_prefix, k_roi, max_side, variants,
        workers, mask_prefix (R2 prefix for the head masks), shard.
Outputs: head.jsonl (one row per specimen: per-variant head_um3, Dc_um, touch, status), qc/<key>.jpg (rows = variants
A/B/C, columns = four sections through the brain centre + a projection of the whole ROI; cyan head, red brain), summary.json; masks in R2
<mask_prefix><key>.npz (roi offset in low-res voxels, iso spacing, packed masks per variant).
"""
import io, json, time, collections, concurrent.futures as cf
import numpy as np
from scipy import ndimage as ndi
import r2

VARIANTS = ("A", "B", "C")


def edt(m, sp): return ndi.distance_transform_edt(m, sampling=sp)


def opening(m, r, sp):
    if r <= 0: return m
    er = edt(m, sp) >= r
    return m & (edt(~er, sp) <= r)


def closing(m, r, sp):
    if r <= 0: return m
    di = edt(~m, sp) <= r                      # dilation by r
    return (edt(di, sp) > r) | m               # then erosion by r


def keep_seeded(m, seed):
    lab, n = ndi.label(m)
    if n == 0: return m
    hit = np.bincount(lab[seed & m], minlength=n + 1); hit[0] = 0
    return lab == int(np.argmax(hit)) if hit.max() > 0 else np.zeros_like(m)


def body_mask(g, seed):
    """Thresholds from high to low (thin cuticle at coarse low-res needs a lower one), closing 1 then 2 voxels; the first
    mask that holds the brain and fills < 60 % of the ROI (no leak into the medium) is used."""
    from skimage.filters import threshold_multiotsu
    t_lo, t_hi = [float(t) for t in threshold_multiotsu(g[::2, ::2, ::2], classes=3)]
    for T in (t_hi, (t_lo + t_hi) / 2, t_lo + 0.25 * (t_hi - t_lo)):
        for it in (1, 2):
            C = ndi.binary_closing(np.pad(g > T, 3), iterations=it)[3:-3, 3:-3, 3:-3]
            F = C.copy()
            for ax in range(3):
                for i in range(C.shape[ax]):
                    sl = [slice(None)] * 3; sl[ax] = i; sl = tuple(sl)
                    F[sl] |= ndi.binary_fill_holes(C[sl])
            F = keep_seeded(ndi.binary_fill_holes(F), seed)
            if F.any() and F.mean() < 0.6: return F, C, T
    return None, None, t_hi


def split_core(F, seed, sp, Dc, D, open_frac):
    from skimage.segmentation import watershed
    core = D >= 0.5 * Dc; lc, _ = ndi.label(core)
    hid = set(np.unique(lc[seed & core])) - {0}
    mk = np.zeros(F.shape, np.int32); mk[np.isin(lc, list(hid)) | (seed & F)] = 1
    mk[core & ~np.isin(lc, list(hid))] = 2
    bnd = np.zeros(F.shape, bool); bnd[[0, -1]] = True; bnd[:, [0, -1]] = True; bnd[:, :, [0, -1]] = True
    mk[bnd & F & (mk == 0)] = 2
    head = watershed(-D, mk, mask=F) == 1
    return keep_seeded(opening(head, open_frac * Dc, sp), seed)


def split_hmax(F, seed, sp, Dc, D, open_frac, h_frac=0.08):
    from skimage.segmentation import watershed
    from skimage.morphology import h_maxima
    core = D >= 0.5 * Dc; lc, _ = ndi.label(core)
    hcore = np.isin(lc, list(set(np.unique(lc[seed & core])) - {0})) | (seed & F)
    hm = h_maxima(np.round(D).astype(np.int32), max(1, int(round(h_frac * Dc)))) > 0
    mk, _ = ndi.label(hm)
    ws = watershed(-D, mk, mask=F)
    head = np.isin(ws, list(set(np.unique(mk[hm & hcore])) - {0})) | hcore
    return keep_seeded(opening(head, open_frac * Dc, sp), seed)


def split_interior(F, C, seed, sp, Dc_body, open_frac, wall_vox=3):
    Cd = ndi.binary_dilation(C, iterations=1)
    I = keep_seeded(F & ~Cd, seed)
    if not I.any(): return None
    D = edt(I, sp); Dc = float(np.percentile(D[seed & I], 90)) if (seed & I).any() else Dc_body
    hi = split_hmax(I, seed, sp, Dc, D, open_frac)
    head = hi | ((edt(~hi, sp) <= wall_vox * sp[0]) & Cd & F)      # the head's own cuticle wall
    head = closing(head, open_frac * Dc_body, sp) & F                 # bridges sclerites inside the head
    return keep_seeded(ndi.binary_fill_holes(head), seed)


def touches(m): return [bool(m[0].any()), bool(m[-1].any()), bool(m[:, 0].any()), bool(m[:, -1].any()), bool(m[:, :, 0].any()), bool(m[:, :, -1].any())]


def measure(v, f, vox, mk, bz, cfg):
    zz, yy, xx = np.nonzero(mk)
    full = np.stack([zz + bz[0], yy + bz[2], xx + bz[4]], 1)
    lo, hi = full.min(0), full.max(0) + 1
    sp0 = vox * f; clr = (lo + hi) / 2 / f; R0 = (hi - lo).max() / 2 * vox
    kr = float(cfg.get("k_roi", 4.0)); res = None
    for attempt in range(3):
        R = kr * R0
        a = np.maximum(0, np.floor(clr - R / sp0)).astype(int); b = np.minimum(v.shape, np.ceil(clr + R / sp0)).astype(int)
        s = max(float(sp0.max()), float(np.max((b - a) * sp0)) / cfg.get("max_side", 220))
        zf = sp0 / s; roi = ndi.zoom(v[a[0]:b[0], a[1]:b[1], a[2]:b[2]], zf, order=1); sp = np.array([s] * 3)
        pts = np.round((full[:: max(1, len(full) // 300000)] / f - a) * zf).astype(int)
        pts = pts[np.all((pts >= 0) & (pts < roi.shape), 1)]
        seed = np.zeros(roi.shape, bool); seed[tuple(pts.T)] = True
        g = ndi.gaussian_filter(roi.astype(np.float32), 0.7)
        F, C, T = body_mask(g, seed)
        if F is None: return {"status": "brain_not_in_body", "T": round(T, 1)}, None
        D = edt(F, sp); Dc = float(np.percentile(D[seed & F], 90))
        heads = {}
        try: heads["A"] = split_core(F, seed, sp, Dc, D, cfg.get("open_frac", 0.15))
        except Exception as ex: heads["A"] = None
        try: heads["B"] = split_hmax(F, seed, sp, Dc, D, cfg.get("open_frac", 0.15))
        except Exception as ex: heads["B"] = None
        try: heads["C"] = split_interior(F, C, seed, sp, Dc, cfg.get("open_frac", 0.15))
        except Exception as ex: heads["C"] = None
        inner = [a[0] > 0, b[0] < v.shape[0], a[1] > 0, b[1] < v.shape[1], a[2] > 0, b[2] < v.shape[2]]
        hA = heads.get("A") if heads.get("A") is not None else next((h for h in heads.values() if h is not None), None)
        bad_touch = hA is not None and any(t and i for t, i in zip(touches(hA), inner))   # grow the ROI for variant A only
        res = (roi, sp, a, zf, seed, heads, T, Dc, kr)
        if not bad_touch: break
        kr *= 1.5
    roi, sp, a, zf, seed, heads, T, Dc, kr = res
    rec = {"status": "ok", "T": round(T, 1), "Dc_um": round(Dc, 1), "k_roi": kr, "iso_um": round(float(sp[0]), 3), "roi_shape": list(roi.shape),
           "brain_seed_um3": float(seed.sum() * np.prod(sp))}
    for k in VARIANTS:
        h = heads.get(k)
        rec[f"{k}_head_um3"] = float(h.sum() * np.prod(sp)) if h is not None else None
        rec[f"{k}_touch"] = touches(h) if h is not None else None
    return rec, (roi, sp, a, zf, seed, heads)


def qc_jpg(roi, seed, heads, path):
    from PIL import Image
    c = np.round(ndi.center_of_mass(seed)).astype(int)
    lo, hi = np.percentile(roi, [0.5, 99.9]); W = 300
    brain = ndi.binary_closing(ndi.binary_dilation(seed, iterations=1), iterations=2)
    ref = heads.get("A") if heads.get("A") is not None else next((h for h in heads.values() if h is not None), seed)
    ez = np.nonzero(ref.any((1, 2)))[0]; zoff = int(c[0] + 0.3 * (ez.max() - c[0])) if len(ez) else c[0]
    cuts = [(0, c[0]), (0, min(zoff, roi.shape[0] - 1)), (1, c[1]), (2, c[2])]
    U = np.zeros(roi.shape, bool)
    for h in heads.values():
        if h is not None: U |= h
    U |= brain
    nz = [np.nonzero(U.any(tuple(j for j in range(3) if j != ax)))[0] for ax in range(3)]
    box = [(max(0, int(n.min() - 0.25 * (n.max() - n.min()) - 3)), min(roi.shape[ax], int(n.max() + 0.25 * (n.max() - n.min()) + 4))) for ax, n in enumerate(nz)]
    rows = []
    for k in VARIANTS:
        h = heads.get(k); tiles = []
        for ax, i in cuts:
            sl = [slice(b0, b1) for b0, b1 in box]; sl[ax] = i; sl = tuple(sl)
            a = np.clip((roi[sl].astype(np.float32) - lo) / max(hi - lo, 1), 0, 1); rgb = np.stack([a] * 3, -1)
            if h is not None:
                hm = h[sl]; rgb[hm & ~ndi.binary_erosion(hm)] = [0.1, 0.9, 1.0]
            bm = brain[sl]; rgb[bm & ~ndi.binary_erosion(bm)] = [1, 0.2, 0.2]
            im = Image.fromarray((rgb * 255).astype(np.uint8)); sc = W / max(im.size)
            im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))))
            t = Image.new("RGB", (W, W), (0, 0, 0)); t.paste(im, (0, 0))
            from PIL import ImageDraw
            ImageDraw.Draw(t).text((4, 4), f"{k}" if len(tiles) == 0 else "", fill=(255, 255, 0))
            tiles.append(np.asarray(t))
        # context: projection of the whole ROI along the axis of the smallest head extent, head outline projected
        ax = int(np.argmin([b1 - b0 for b0, b1 in box]))
        pr = roi.max(ax).astype(np.float32); a = np.clip((pr - lo) / max(hi - lo, 1), 0, 1); rgb = np.stack([a] * 3, -1) * 0.8
        if h is not None:
            hp = h.any(ax); rgb[hp & ~ndi.binary_erosion(hp)] = [0.1, 0.9, 1.0]
        bp = brain.any(ax); rgb[bp & ~ndi.binary_erosion(bp)] = [1, 0.2, 0.2]
        im = Image.fromarray((rgb * 255).astype(np.uint8)); sc = W / max(im.size)
        im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))))
        t = Image.new("RGB", (W, W), (0, 0, 0)); t.paste(im, (0, 0)); tiles.append(np.asarray(t))
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path, quality=85)


def one(args):
    key, mkey, cfg, out, tp = args
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read(); t0 = time.time()
    try:
        if key not in tp: return {"key": key, "status": "no_true_pages"}
        z = np.load(io.BytesIO(get(cfg.get("cache_prefix", "ant-twin/cache/lowres/") + key + ".npz")))
        v, nlow, H, W, vox = z["img"], int(z["n_lowres"]), int(z["H"]), int(z["W"]), float(z["voxel_um"])
        P = int(tp[key]["pages"]); f = np.array([P / nlow, H / v.shape[1], W / v.shape[2]])
        m = np.load(io.BytesIO(get(mkey))); mk = m["mask"]
        if mk.sum() == 0: return {"key": key, "status": "empty_brain_mask"}
        rec, extra = measure(v, f, vox, mk, m["box_zyx"], cfg)
        rec = {"key": key, "brain_mask": mkey, "voxel_um": vox, **rec}
        if extra is not None:
            roi, sp, a, zf, seed, heads = extra
            qc_jpg(roi, seed, heads, out / "qc" / f"{key}.jpg")
            buf = io.BytesIO()
            np.savez_compressed(buf, roi_offset_lowres=a, zoom=zf, iso_um=sp[0], shape=np.array(roi.shape),
                                **{k: np.packbits(h) for k, h in heads.items() if h is not None})
            c.put_object(Bucket=r2.BUCKET, Key=cfg.get("mask_prefix", "ant-twin/cache/head_masks_v1/") + key + ".npz", Body=buf.getvalue())
        rec["sec"] = round(time.time() - t0, 1)
        return rec
    except Exception as ex:
        return {"key": key, "status": "error", "error": repr(ex)[:300], "sec": round(time.time() - t0, 1)}


def main(cfg, out):
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    items = json.loads(get(cfg["items_key"])); tp = json.loads(get(cfg["true_pages_key"]))
    si, sn = cfg.get("shard", [0, 1]); items = items[si::sn]
    (out / "qc").mkdir(exist_ok=True)
    print(f"specimens {len(items)} (shard {si}/{sn})", flush=True)
    rows = []
    with cf.ProcessPoolExecutor(cfg.get("workers", 8)) as ex, open(out / "head.jsonl", "w", encoding="utf-8") as fo:
        futs = [ex.submit(one, (k, mk, cfg, out, tp)) for k, mk in items]
        for fu in cf.as_completed(futs):
            rec = fu.result(); rows.append(rec); fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec)[:300], flush=True)
    summ = {"n": len(rows), "status": dict(collections.Counter(r["status"] for r in rows))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
