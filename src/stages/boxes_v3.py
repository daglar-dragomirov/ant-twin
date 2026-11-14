"""Stage boxes_v3: full-resolution brain boxes for every specimen the low-res finder located.

Reads finder.jsonl (box_zyx in full-res page/row/col coordinates), fetches each box once from the KIT server — HTTP
Range on uncompressed BigTIFFs, whole-file download for compressed classic TIFFs — and stores it in R2 as
ant-twin/cache/boxes_v3/<key>.npz (box uint8 z,y,x + box_zyx + voxel_um). Resume-safe; bounded connections.
Config: finder_run, workers, max_box_mb, prefix.
Outputs: boxes_index.jsonl, summary.json.
"""
import io, json, time, tempfile, os, shutil, pathlib, urllib.request, concurrent.futures as cf
import numpy as np
import r2
import bigtiff_range as BT

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}


def main(cfg, out):
    c = r2.client(); pre = cfg.get("prefix", "ant-twin/cache/boxes_v3/")
    runs = cfg.get("finder_runs") or [cfg["finder_run"]]; seen = {}  # several finder runs (e.g. two pods from both ends) merged by key
    for run in runs:
        try: txt = c.get_object(Bucket=r2.BUCKET, Key=f"ant-twin/runs/{run}/outputs/finder.jsonl")["Body"].read().decode()
        except Exception as ex: print("no finder output yet for", run, ex, flush=True); continue
        for l in txt.splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "ok" and r["key"] not in seen: seen[r["key"]] = r
    rows = list(seen.values())
    if cfg.get("reverse"): rows = rows[::-1]  # second pod walks the list from the other end
    have = set(r2.list_keys(c, r2.BUCKET, pre)); print(f"finder ok: {len(rows)}, cached boxes: {len(have)}", flush=True)

    def one(r):
        key = r["key"]; k = pre + key + ".npz"
        if k in have: return {"key": key, "status": "cached"}
        t0 = time.time(); z0, z1, y0, y1, x0, x1 = r["box_zyx"]; url = r["url_full"]
        mb = (z1 - z0) * (y1 - y0) * (x1 - x0) / 1e6
        if mb > cfg.get("max_box_mb", 1500): return {"key": key, "status": "box_too_big", "mb": round(mb)}
        try:
            info = BT.layout(url)
            if info["compression"] == 1:
                if not BT.verify_offset(url, info, z1 - 1): return {"key": key, "status": "layout_irregular"}
                box = BT.crop(url, info, z0, z1, y0, y1, x0, x1); how = "range"
            else:
                import tifffile
                tmp = pathlib.Path(tempfile.gettempdir()) / f"bx3_{key}.tif"
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1200) as resp, open(tmp, "wb") as f: shutil.copyfileobj(resp, f, 1 << 22)
                with tifffile.TiffFile(tmp) as tf: z1 = min(z1, len(tf.pages)); box = np.stack([tf.pages[i].asarray()[y0:y1, x0:x1] for i in range(z0, z1)])
                os.remove(tmp); how = "full_file"
            buf = io.BytesIO(); np.savez_compressed(buf, box=box, box_zyx=np.array([z0, z1, y0, y1, x0, x1]), voxel_um=float(r["voxel_um"]))
            c.put_object(Bucket=r2.BUCKET, Key=k, Body=buf.getvalue())
            return {"key": key, "status": "ok", "fetch": how, "shape": list(box.shape), "mb": round(box.nbytes / 1e6, 1), "npz_mb": round(buf.tell() / 1e6, 1), "sec": round(time.time() - t0, 1)}
        except Exception as ex:
            return {"key": key, "status": "error", "error": repr(ex)[:200]}

    n = 0
    with cf.ThreadPoolExecutor(cfg.get("workers", 4)) as ex, open(out / "boxes_index.jsonl", "w", encoding="utf-8") as f:
        for rec in ex.map(one, rows):
            n += 1; f.write(json.dumps(rec) + "\n"); f.flush()
            if rec["status"] != "cached": print(n, json.dumps(rec), flush=True)
    recs = [json.loads(l) for l in open(out / "boxes_index.jsonl", encoding="utf-8")]
    from collections import Counter
    summ = {"n": len(recs), "status": dict(Counter(r["status"] for r in recs))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
