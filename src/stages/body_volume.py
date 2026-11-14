"""Stage body_volume: whole-body volume of every colony-size worker from the R2 low-res cache, validated against the
owners' S1 body volumes (Matte et al. 2025; their whole-body network) on the overlapping specimens.

Segmentation of a low-res stack: Gaussian smoothing, Otsu threshold, 3D closing, per-slice hole filling, then the
largest 3D component whose mean per-slice solidity is high enough (rejects the thin sample-tube ring). Volume uses the
physical voxel (full-res voxel x full/low-res scale per axis).
Config: specimens_run, owners_s1_key (R2 key of Dataset_S1 xlsx), cache_prefix, workers, qc_every.
Outputs: body.jsonl, validation.json (overlap with S1), qc/<key>.png, summary.json.
"""
import io, json, concurrent.futures as cf
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import r2


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def otsu(x):
    h, e = np.histogram(x, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    wt, mt = w[-1], mu[-1]; w1 = wt - w; ok = (w > 0) & (w1 > 0)
    sb = np.zeros_like(c); sb[ok] = (mt * w[ok] / wt - mu[ok]) ** 2 * wt / (w[ok] * w1[ok]); return c[np.argmax(sb)]


def segment(v):
    from skimage.filters import threshold_multiotsu
    from skimage.morphology import disk
    g = ndi.gaussian_filter(v.astype(np.float32), 1.0)
    t0, t1 = (float(x) for x in threshold_multiotsu(g[::2, ::4, ::4], classes=3))  # dark outside | medium | cuticle-bright
    se = disk(3); body = np.zeros(g.shape, bool)
    for z in range(g.shape[0]):
        tube = ndi.binary_fill_holes(g[z] > t0)                                  # tube interior incl. wall
        inner = ndi.binary_erosion(tube, structure=disk(1), iterations=6)        # cut the wall (+ margin)
        sh = (g[z] > t1) & inner                                                 # cuticle inside the tube
        sh = ndi.binary_closing(sh, structure=se)                                # bridge small gaps in the shell
        body[z] = ndi.binary_fill_holes(sh)
    lab, n = ndi.label(body)
    if n == 0: return body, {"n_comp": 0, "thresholds": [t0, t1]}
    sizes = ndi.sum(body, lab, range(1, n + 1)); keep = [i + 1 for i in range(n) if sizes[i] >= 0.03 * sizes.max()]
    return np.isin(lab, keep), {"n_comp": int(n), "n_kept": len(keep), "thresholds": [round(t0, 2), round(t1, 2)]}


def main(cfg, out):
    import pandas as pd
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    d = pd.read_csv(io.BytesIO(get(f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")))
    d = d[(d.caste == "worker") & d.colony_any.notna()]
    s1 = pd.read_excel(io.BytesIO(get(cfg["owners_s1_key"])))
    s1b = dict(zip(s1.specimen_code.astype(str), s1.body_volume))
    have = set(r2.list_keys(c, r2.BUCKET, cfg["cache_prefix"])); (out / "qc").mkdir(exist_ok=True)

    def one(ir):
        i, r = ir; key = f"{r['folder']}__{vial_code(r['vial'])}"; k = cfg["cache_prefix"] + key + ".npz"
        rec = {"key": key, "specimen_code": r["specimen_code"], "species_key": r["species_key"], "stained": r["stained"], "voxel_um": r["voxel_um"],
               "s1_body_um3": float(s1b[r["specimen_code"]]) if r["specimen_code"] in s1b else None}
        if k not in have: rec["status"] = "not_cached"; return rec
        try:
            z = np.load(io.BytesIO(get(k))); v = z["img"]; vox = float(z["voxel_um"])
            sp = (vox * int(z["pages"]) / int(z["n_lowres"]), vox * int(z["H"]) / v.shape[1], vox * int(z["W"]) / v.shape[2])
            m, info = segment(v); vol = float(m.sum() * sp[0] * sp[1] * sp[2])
            rec.update(status="ok", body_um3=vol, spacing_um=[round(x, 3) for x in sp], **info)
            if rec["s1_body_um3"]: rec["ratio_to_s1"] = round(vol / rec["s1_body_um3"], 4)
            if i % cfg.get("qc_every", 25) == 0:
                zc = int(np.median(np.nonzero(m.any((1, 2)))[0])); a = v[zc].astype(np.float32); a = (a - a.min()) / max(np.ptp(a), 1)
                rgb = np.stack([a] * 3, -1); e = m[zc] & ~ndi.binary_erosion(m[zc]); rgb[e] = [1, .2, .2]
                Image.fromarray((rgb * 255).astype(np.uint8)).save(out / "qc" / f"{key}.png")
        except Exception as ex:
            rec.update(status="error", error=repr(ex)[:200])
        return rec

    recs = []
    with cf.ThreadPoolExecutor(cfg.get("workers", 16)) as ex, open(out / "body.jsonl", "w", encoding="utf-8") as f:
        for rec in ex.map(one, enumerate(d.to_dict("records"))):
            recs.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
    ov = [r for r in recs if r.get("status") == "ok" and r.get("s1_body_um3")]
    val = {}
    if len(ov) >= 5:
        x = np.log10([r["s1_body_um3"] for r in ov]); y = np.log10([r["body_um3"] for r in ov])
        b, a = np.polyfit(x, y, 1)
        val = {"n_overlap": len(ov), "pearson_r_log": float(np.corrcoef(x, y)[0, 1]), "slope_log": float(b), "intercept_log": float(a),
               "median_ratio": float(np.median([r["ratio_to_s1"] for r in ov])), "ratio_p10_p90": [float(np.percentile([r["ratio_to_s1"] for r in ov], q)) for q in (10, 90)]}
    (out / "validation.json").write_text(json.dumps(val, indent=1))
    from collections import Counter
    summ = {"n": len(recs), "status": dict(Counter(r["status"] for r in recs)), "with_s1": sum(r["s1_body_um3"] is not None for r in recs), "validation": val}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
