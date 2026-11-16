"""Stage lowres_cache: fetch every needed AntScan low-res stack once and keep it in R2.

Politeness to the KIT server: at most `workers` specimens at a time, each with 8 HTTP connections. Resume-safe: keys
already in the cache are skipped. Each entry is ant-twin/cache/lowres/<folder>__<vial>.npz holding `img` (uint8 z,y,x)
and geometry (n_lowres, full-res pages/H/W, voxel_um, file_bytes).
Config: specimens_run, workers, only_workers_with_colony, max_specimens, prefix.
Outputs: cache_index.jsonl (key, status, shape, bytes, seconds), summary.json.
"""
import io, json, time, tempfile, os, pathlib, concurrent.futures as cf
import numpy as np
import r2
import bigtiff_range as BT
from stages.lowres_finder_data import lowres_stack


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def main(cfg, out):
    import pandas as pd
    c = r2.client(); pre = cfg.get("prefix", "ant-twin/cache/lowres/")
    d = pd.read_csv(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")["Body"].read()))
    d = d[(d.caste == "worker") & d.url_full.notna() & d.url_lowres_slice0.notna()]
    if cfg.get("only_workers_with_colony", True): d = d[d.colony_any.notna()]
    recs = d.to_dict("records")[: cfg.get("max_specimens", 100000)]
    have = set(r2.list_keys(c, r2.BUCKET, pre))
    print(f"specimens {len(recs)}, already cached {len(have)}", flush=True)

    def one(r):
        key = f"{r['folder']}__{vial_code(r['vial'])}"; k = pre + key + ".npz"
        if k in have: return {"key": key, "status": "cached"}
        t0 = time.time()
        try:
            v, nlow = lowres_stack(r["url_lowres_slice0"].replace("slice_0000.png", "slice_%04d.png"))
            if v is None: return {"key": key, "status": "no_lowres", "n_lowres": nlow}
            info = BT.layout(r["url_full"]); size, P = BT.n_pages(r["url_full"], info)
            buf = io.BytesIO(); np.savez_compressed(buf, img=v, n_lowres=nlow, pages=P, H=info["H"], W=info["W"], voxel_um=float(r["voxel_um"]), file_bytes=size)
            c.put_object(Bucket=r2.BUCKET, Key=k, Body=buf.getvalue())
            return {"key": key, "status": "ok", "shape": list(v.shape), "bytes": buf.tell(), "pages": P, "sec": round(time.time() - t0, 1)}
        except Exception as ex:
            return {"key": key, "status": "error", "error": repr(ex)[:200]}

    n = 0
    with cf.ThreadPoolExecutor(cfg.get("workers", 4)) as ex, open(out / "cache_index.jsonl", "w", encoding="utf-8") as f:
        for rec in ex.map(one, recs):
            n += 1; f.write(json.dumps(rec) + "\n"); f.flush()
            if rec["status"] != "cached": print(n, json.dumps(rec), flush=True)
    rows = [json.loads(l) for l in open(out / "cache_index.jsonl", encoding="utf-8")]
    from collections import Counter
    summ = {"n": len(rows), "status": dict(Counter(r["status"] for r in rows)), "gb_new": round(sum(r.get("bytes", 0) for r in rows) / 1e9, 2)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
