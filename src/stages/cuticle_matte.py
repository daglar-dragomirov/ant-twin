"""Stage cuticle_matte: body, cuticle and soft-tissue volume with the authors' published segmentation program.

Why: robustness check R3 (brain relative to soft tissue) needs body minus cuticle for every species; the owners' S1
(Matte et al. 2025) gives it only for part of our specimens. Our own threshold pipeline (cuticle_fullres v1) leaked on
bright-medium scans, so we run the authors' program itself (vendored unchanged, src/vendor/matte2025) on the full-res
AntScan stacks. On the S1 overlap this must reproduce S1; that agreement is the validation.
Per specimen (as the authors' auto_multiscan_segmentation + extract_data_segmentation):
1. download the full-res 8-bit stack (url_full) to local disk; true page count must match meta/true_pages.json;
2. beginning_and_end() (authors' specimen z-limits, every 30th slice);
3. every seg_image_gap-th slice in [beginning, end): color_image_processing -> remove_vial -> segmente_image(threshold)
   -> filter_segmented_image(merge_contour) -> fill_contours; body = white pixels of the filled image, cuticle = white
   pixels of the filtered contour image (authors' get_metrics);
4. per-slice series -> scipy median_filter(size=25) -> sum (authors' compile_bv_and_cv, single-scan case);
   volume = voxel count x voxel_um^3 x seg_image_gap.
Only file I/O differs from the authors' scripts: slices are read from one open TiffFile instead of re-opening the file.
Config: specimens_run, keys_key (optional R2 list of keys), true_pages_key, thresholds (list; authors' scripts use
[110, 255] (single scan) and [145, 255] (multiscan); each threshold gets its own limits and volumes, prefix t<low>_), merge_contour, vial_effort, seg_image_gap, workers, cv_threads, shard, qc_slices, tmp_dir.
Outputs: cuticle.jsonl, slices/<key>.npz (per-slice raw counts), qc/<key>.jpg, summary.json.
"""
import io, json, os, time, shutil, pathlib, urllib.request, concurrent.futures as cf
import numpy as np
import r2

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def _load_matte(cv_threads):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import cv2
    cv2.setNumThreads(int(cv_threads))
    from vendor.matte2025 import internal__auto_scan_segmentation as M
    return M


def segment_specimen(path, cfg):
    """Authors' per-slice chain on one local stack; returns per-slice body/cuticle pixel counts and limits."""
    import tifffile
    M = _load_matte(cfg.get("cv_threads", 2))
    tf = tifffile.TiffFile(path)
    try:
        pages = tf.pages
        # I/O only: the authors' helpers re-open the file per slice; read from the open file instead
        M.open_single_img_tif = lambda file_path, i: pages[i].asarray()
        M.count_2d_images = lambda file_path: len(pages)
        M.tqdm = lambda it, **k: it
        ths = [list(t) for t in cfg.get("thresholds", [[145, 255]])]; ve = cfg.get("vial_effort", False); mc = bool(cfg.get("merge_contour", False))
        gap = int(cfg.get("seg_image_gap", 1)); res = {}
        lim = {tuple(t): M.beginning_and_end(path, [t, mc, ve, True]) for t in ths}
        lo = min(v[0] for v in lim.values()); hi = max(v[1] for v in lim.values())
        bv = {tuple(t): [] for t in ths}; cv = {tuple(t): [] for t in ths}; qc = []
        idx = list(range(lo, hi, gap))
        qsel = {idx[j] for j in np.linspace(0, len(idx) - 1, cfg.get("qc_slices", 4)).astype(int)} if idx else set()
        qt = tuple(ths[0])
        for i in idx:
            image = pages[i].asarray()
            adj = M.color_image_processing(image) if not ve else M.color_image_processing_2(image)
            wv = M.remove_vial(adj)
            for t in ths:
                tt = tuple(t); b0, e0 = lim[tt]
                if not (b0 <= i < e0): bv[tt].append(np.nan); cv[tt].append(np.nan); continue
                sg = M.segmente_image(wv, t)
                if ve: sg = M.remove_vial_2(sg, ve)
                filt = M.filter_segmented_image(sg, mc)
                neg = M.fill_contours(filt)
                bv[tt].append(M.get_metrics(neg)); cv[tt].append(M.get_metrics(filt))
                if i in qsel and tt == qt: qc.append((image, filt, neg))
        out = {}
        for t in ths:
            tt = tuple(t); b_ = np.array(bv[tt], float); c_ = np.array(cv[tt], float); ok = ~np.isnan(b_)
            out[tt] = (lim[tt], b_[ok], c_[ok])
        return len(pages), out, qc
    finally:
        tf.close()


def qc_jpg(qc, path):
    from PIL import Image
    rows = []
    for image, filt, neg in qc:
        a = image.astype(np.float32); a = (a - a.min()) / max(np.ptp(a), 1)
        rgb = np.stack([a] * 3, -1)
        rgb[(neg > 0) & ~(filt > 0)] = rgb[(neg > 0) & ~(filt > 0)] * 0.6 + np.array([0, 0, 0.4])   # body (soft) blue
        rgb[filt > 0] = rgb[filt > 0] * 0.4 + np.array([0, 0.6, 0])                                   # cuticle green
        rows.append(np.concatenate([np.stack([a] * 3, -1), rgb], 1))
    if not rows: return
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    im = Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)); im.thumbnail((1400, 2800)); im.save(path, quality=85)


def one(args):
    r, cfg, out, tp = args
    from scipy.ndimage import median_filter
    key = r["key"]; t0 = time.time(); vox = float(r["voxel_um"])
    base = {"key": key, "specimen_code": r.get("specimen_code"), "voxel_um": vox}
    tmp = pathlib.Path(cfg.get("tmp_dir", "/root/tmp_cut")) / f"{key}.tif"; tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        if key not in tp: return {**base, "status": "no_true_pages"}
        with urllib.request.urlopen(urllib.request.Request(r["url_full"], headers=UA), timeout=3600) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, 1 << 22)
        t_dl = time.time() - t0
        n_pages, res, qc = segment_specimen(str(tmp), cfg)
        if n_pages != int(tp[key]["pages"]): return {**base, "status": "page_mismatch", "n_pages": n_pages, "pages_true": int(tp[key]["pages"])}
        gap = int(cfg.get("seg_image_gap", 1)); v3 = vox ** 3 * gap; rec = {**base, "status": "ok", "n_pages": n_pages, "gap": gap}; arrs = {}
        for tt, ((beg, end), bv, cv) in res.items():
            tag = f"t{tt[0]}"
            bvf = median_filter(bv, size=25) if len(bv) else bv; cvf = median_filter(cv, size=25) if len(cv) else cv
            b, c = float(bvf.sum() * v3), float(cvf.sum() * v3)
            rec.update({f"{tag}_beginning": int(beg), f"{tag}_end": int(end), f"{tag}_n_slices": int(len(bv)),
                        f"{tag}_body_um3": b, f"{tag}_cuticle_um3": c, f"{tag}_soft_um3": b - c,
                        f"{tag}_body_um3_raw": float(bv.sum() * v3), f"{tag}_cuticle_um3_raw": float(cv.sum() * v3)})
            arrs[f"{tag}_body"] = bv.astype(np.int64); arrs[f"{tag}_cuticle"] = cv.astype(np.int64); arrs[f"{tag}_beginning"] = beg
        np.savez_compressed(out / "slices" / f"{key}.npz", gap=gap, **arrs)
        if qc: qc_jpg(qc, out / "qc" / f"{key}.jpg")
        return {**rec, "sec_download": round(t_dl, 1), "sec": round(time.time() - t0, 1)}
    except Exception as ex:
        return {**base, "status": "error", "error": repr(ex)[:300], "sec": round(time.time() - t0, 1)}
    finally:
        if tmp.exists(): os.remove(tmp)


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
    tp = json.loads(get(cfg["true_pages_key"]))
    (out / "qc").mkdir(exist_ok=True); (out / "slices").mkdir(exist_ok=True)
    print(f"specimens {len(recs)} (shard {si}/{sn})", flush=True)
    rows = []
    with cf.ProcessPoolExecutor(cfg.get("workers", 8)) as ex, open(out / "cuticle.jsonl", "w", encoding="utf-8") as fo:
        futs = [ex.submit(one, (r, cfg, out, tp)) for r in recs]
        for f in cf.as_completed(futs):
            rec = f.result(); rows.append(rec)
            fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec)[:300], flush=True)
    from collections import Counter
    summ = {"n": len(rows), "status": dict(Counter(r["status"] for r in rows))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
