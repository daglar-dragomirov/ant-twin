"""Stage boxes_grow: re-cut brain boxes whose mask touches a box face (brain cut off by the box border).

For every key reviewed with box_on_head == true (later review sets override earlier ones) whose mask (mask_prefixes, first
hit wins) touches one or more faces of its box, each touching face is moved outward by grow_frac x the box extent on
that axis, clipped to the true stack size (meta/true_pages.json), and the box is re-cut from the full-res stack into
cache/boxes_grow/<key>.npz (box, box_zyx, voxel_um, grown_faces). Uncompressed stacks: HTTP Range crop; compressed:
whole-file download. Tiny-ant end keys (key@A/@B) use the base key's stack.
Config: reviews_unchanged_only [files], reviews [files], box_prefixes, mask_prefixes, true_pages_key, specimens_run, prefix, grow_frac, workers, max_box_mb.
Outputs: boxes_index.jsonl, summary.json.
"""
import io, json, time, tempfile, os, shutil, pathlib, urllib.request, collections, concurrent.futures as cf
import numpy as np
import r2
import bigtiff_range as BT

def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}


def touching(m):
    return [bool(m[0].any()), bool(m[-1].any()), bool(m[:, 0].any()), bool(m[:, -1].any()), bool(m[:, :, 0].any()), bool(m[:, :, -1].any())]


def main(cfg, out):
    import pandas as pd
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    tp = json.loads(get(cfg["true_pages_key"]))
    d = pd.read_csv(io.BytesIO(get(f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")))
    url = {f"{r['folder']}__{vial_code(r['vial'])}": r["url_full"] for r in d.to_dict("records") if isinstance(r.get("url_full"), str)}
    unchanged = lambda k: tp.get(k.split("@")[0], {}).get("pages") == tp.get(k.split("@")[0], {}).get("pages_est")
    rev = {}
    for f in cfg.get("reviews_unchanged_only", []):  # reviews of boxes_v3 boxes: valid only where the page estimate was right
        for x in json.loads(get(f)):
            if unchanged(x["key"]): rev[x["key"]] = x
    for f in cfg["reviews"]:
        for x in json.loads(get(f)): rev[x["key"]] = x
    onhead = sorted(k for k, x in rev.items() if x.get("box_on_head") is True)
    listing = {p: {k[len(p):-4] for k in r2.list_keys(c, r2.BUCKET, p)} for p in cfg["box_prefixes"] + cfg["mask_prefixes"]}
    pre = cfg.get("prefix", "ant-twin/cache/boxes_grow/"); have = {k[len(pre):-4] for k in r2.list_keys(c, r2.BUCKET, pre)}
    print(f"reviewed {len(rev)}, on head {len(onhead)}, already grown {len(have)}", flush=True)
    g = float(cfg.get("grow_frac", 0.5))

    def one(key):
        if key in have: return {"key": key, "status": "cached"}
        bp = next((p for p in cfg["box_prefixes"] if key in listing[p]), None); mp = next((p for p in cfg["mask_prefixes"] if key in listing[p]), None)
        if bp is None or mp is None: return {"key": key, "status": "no_box_or_mask"}
        base = key.split("@")[0]
        if base not in tp or base not in url: return {"key": key, "status": "no_geometry"}
        t0 = time.time()
        try:
            mz = np.load(io.BytesIO(get(mp + key + ".npz"))); m = mz["mask"].astype(bool)
            bz = np.load(io.BytesIO(get(bp + key + ".npz"))); z0, z1, y0, y1, x0, x1 = [int(a) for a in bz["box_zyx"]]; vox = float(bz["voxel_um"])
            if m.sum() == 0: return {"key": key, "status": "empty_mask"}
            tch = touching(m)
            if not any(tch): return {"key": key, "status": "not_touching"}
            lim = [int(tp[base]["pages"]), int(tp[base]["H"]), int(tp[base]["W"])]
            b = [[z0, z1], [y0, y1], [x0, x1]]
            for ax in range(3):
                ext = b[ax][1] - b[ax][0]
                if tch[2 * ax]: b[ax][0] = max(0, int(b[ax][0] - g * ext))
                if tch[2 * ax + 1]: b[ax][1] = min(lim[ax], int(b[ax][1] + g * ext))
            (z0, z1), (y0, y1), (x0, x1) = b
            mb = (z1 - z0) * (y1 - y0) * (x1 - x0) / 1e6
            if mb > cfg.get("max_box_mb", 2000): return {"key": key, "status": "box_too_big", "mb": round(mb)}
            u = url[base]; info = BT.layout(u)
            if info["compression"] == 1:
                if not BT.verify_offset(u, info, z1 - 1): return {"key": key, "status": "layout_irregular"}
                box = BT.crop(u, info, z0, z1, y0, y1, x0, x1); how = "range"
            else:
                import tifffile
                tmp = pathlib.Path(tempfile.gettempdir()) / f"bxg_{key.replace('@', '_')}.tif"
                try:
                    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=1200) as resp, open(tmp, "wb") as f: shutil.copyfileobj(resp, f, 1 << 22)
                    with tifffile.TiffFile(tmp) as tf:
                        if len(tf.pages) != lim[0]: return {"key": key, "status": "page_mismatch", "pages_file": len(tf.pages), "pages_true": lim[0]}
                        box = np.stack([tf.pages[i].asarray()[y0:y1, x0:x1] for i in range(z0, z1)])
                finally:
                    if tmp.exists(): os.remove(tmp)
                how = "full_file"
            buf = io.BytesIO(); np.savez_compressed(buf, box=box, box_zyx=np.array([z0, z1, y0, y1, x0, x1]), voxel_um=vox, grown_faces=np.array(tch))
            c.put_object(Bucket=r2.BUCKET, Key=pre + key + ".npz", Body=buf.getvalue())
            return {"key": key, "status": "grown", "faces": tch, "box_zyx": [z0, z1, y0, y1, x0, x1], "fetch": how, "src_box": bp, "src_mask": mp, "sec": round(time.time() - t0, 1)}
        except Exception as ex:
            return {"key": key, "status": "error", "error": repr(ex)[:200]}

    with cf.ThreadPoolExecutor(cfg.get("workers", 6)) as ex, open(out / "boxes_index.jsonl", "w", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(one, onhead)):
            f.write(json.dumps(rec) + "\n"); f.flush()
            if rec["status"] not in ("cached", "not_touching"): print(i, json.dumps(rec), flush=True)
    recs = [json.loads(l) for l in open(out / "boxes_index.jsonl", encoding="utf-8")]
    summ = {"n": len(recs), "status": dict(collections.Counter(r["status"] for r in recs))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
