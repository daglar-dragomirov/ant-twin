"""Stage boxes_v4: brain boxes with the true full-res page count (fix for the compressed-stack P bug).

lowres_cache estimated full-res pages from file size, which is wrong for deflate-compressed stacks, so every finder box
on such a stack had its z range scaled by P_est/n_lowres instead of P_true/n_lowres (y/x were right).
For each key in boxes_v3 plus keys whose boxes_v3 cut failed:
  * P_true == P_est  -> server-side copy boxes_v3 -> boxes_v4 and brain_masks_v2 -> brain_masks_v4 (nothing changed);
  * P_true != P_est  -> take the finder row that produced the boxes_v3 box (same y/x), rescale its unclipped z range by
    P_true/P_est, clip to [0, P_true], re-cut from the full file into boxes_v4 (box_zyx in true page indices).
Tiny-ant end keys (key@A/@B) use the base key's page table.
Config: finder_runs, true_pages_key, src_prefix, prefix, src_mask_prefix, mask_prefix, workers, max_box_mb.
Outputs: boxes_index.jsonl, summary.json.
"""
import io, json, time, tempfile, os, shutil, pathlib, urllib.request, collections, concurrent.futures as cf
import numpy as np
import r2
import bigtiff_range as BT

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}


def main(cfg, out):
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    tp = json.loads(get(cfg.get("true_pages_key", "ant-twin/meta/true_pages.json")))
    spre, pre = cfg.get("src_prefix", "ant-twin/cache/boxes_v3/"), cfg.get("prefix", "ant-twin/cache/boxes_v4/")
    smpre, mpre = cfg.get("src_mask_prefix", "ant-twin/cache/brain_masks_v2/"), cfg.get("mask_prefix", "ant-twin/cache/brain_masks_v4/")
    rows = collections.defaultdict(list)
    for run in cfg["finder_runs"]:
        try: txt = get(f"ant-twin/runs/{run}/outputs/finder.jsonl").decode()
        except Exception as ex: print("no finder output for", run, ex, flush=True); continue
        for l in txt.splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "ok" and "box_zyx" in r: rows[r["key"]].append(r)
    src = {k[len(spre):-4] for k in r2.list_keys(c, r2.BUCKET, spre)}
    smasks = {k[len(smpre):-4] for k in r2.list_keys(c, r2.BUCKET, smpre)}
    have = {k[len(pre):-4] for k in r2.list_keys(c, r2.BUCKET, pre)}
    keys = sorted(src | set(rows))
    if cfg.get("reverse"): keys = keys[::-1]  # second pod walks the list from the other end
    print(f"finder keys {len(rows)}, boxes_v3 {len(src)}, masks_v2 {len(smasks)}, boxes_v4 already {len(have)}, todo {len(keys)}", flush=True)

    def one(key):
        if key in have: return {"key": key, "status": "cached"}
        base = key.split("@")[0]; t = tp.get(base)
        if t is None: return {"key": key, "status": "no_true_pages"}
        Pt, Pe = int(t["pages"]), int(t["pages_est"])
        t0 = time.time()
        try:
            if Pt == Pe and key in src:
                c.copy_object(Bucket=r2.BUCKET, Key=pre + key + ".npz", CopySource={"Bucket": r2.BUCKET, "Key": spre + key + ".npz"})
                m = key in smasks
                if m: c.copy_object(Bucket=r2.BUCKET, Key=mpre + key + ".npz", CopySource={"Bucket": r2.BUCKET, "Key": smpre + key + ".npz"})
                return {"key": key, "status": "copied", "mask_copied": m}
            cand = rows.get(key, [])
            if not cand: return {"key": key, "status": "no_finder_row"}
            r = cand[0]
            if key in src:  # the row that produced the boxes_v3 box has the same y/x
                bz = [int(a) for a in np.load(io.BytesIO(get(spre + key + ".npz")))["box_zyx"]]
                r = next((x for x in cand if list(x["box_zyx"][2:]) == bz[2:]), cand[0])
            z0, z1, y0, y1, x0, x1 = [int(a) for a in r["box_zyx"]]
            s = Pt / Pe; z0, z1 = max(0, int(np.floor(z0 * s))), min(Pt, int(np.ceil(z1 * s)))
            if z1 - z0 < 4: return {"key": key, "status": "empty_z", "z": [z0, z1]}
            mb = (z1 - z0) * (y1 - y0) * (x1 - x0) / 1e6
            if mb > cfg.get("max_box_mb", 1500): return {"key": key, "status": "box_too_big", "mb": round(mb)}
            url = r["url_full"]; info = BT.layout(url)
            if info["compression"] == 1:
                if not BT.verify_offset(url, info, z1 - 1): return {"key": key, "status": "layout_irregular"}
                box = BT.crop(url, info, z0, z1, y0, y1, x0, x1); how = "range"
            else:
                import tifffile
                tmp = pathlib.Path(tempfile.gettempdir()) / f"bx4_{key.replace('@', '_')}.tif"
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1200) as resp, open(tmp, "wb") as f: shutil.copyfileobj(resp, f, 1 << 22)
                    with tifffile.TiffFile(tmp) as tf:
                        n = len(tf.pages)
                        if n != Pt: return {"key": key, "status": "page_mismatch", "pages_file": n, "pages_true": Pt}
                        box = np.stack([tf.pages[i].asarray()[y0:y1, x0:x1] for i in range(z0, z1)])
                finally:
                    if tmp.exists(): os.remove(tmp)
                how = "full_file"
            buf = io.BytesIO(); np.savez_compressed(buf, box=box, box_zyx=np.array([z0, z1, y0, y1, x0, x1]), voxel_um=float(r["voxel_um"]), pages_true=Pt, pages_est=Pe)
            c.put_object(Bucket=r2.BUCKET, Key=pre + key + ".npz", Body=buf.getvalue())
            return {"key": key, "status": "recut", "fetch": how, "scale": round(s, 4), "box_zyx": [z0, z1, y0, y1, x0, x1], "shape": list(box.shape), "sec": round(time.time() - t0, 1)}
        except Exception as ex:
            return {"key": key, "status": "error", "error": repr(ex)[:200]}

    n = 0
    with cf.ThreadPoolExecutor(cfg.get("workers", 4)) as ex, open(out / "boxes_index.jsonl", "w", encoding="utf-8") as f:
        for rec in ex.map(one, keys):
            n += 1; f.write(json.dumps(rec) + "\n"); f.flush()
            if rec["status"] not in ("cached", "copied"): print(n, json.dumps(rec), flush=True)
    recs = [json.loads(l) for l in open(out / "boxes_index.jsonl", encoding="utf-8")]
    summ = {"n": len(recs), "status": dict(collections.Counter(r["status"] for r in recs)), "masks_copied": sum(bool(r.get("mask_copied")) for r in recs)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
