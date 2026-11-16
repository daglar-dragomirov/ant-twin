"""Stage true_pages: exact full-res page count for every cached AntScan stack.

Why: lowres_cache estimated pages as (file_size - data0) // data_stride. That holds only for uncompressed tifs; for
deflate-compressed stacks the per-page size varies and the estimate is wrong by 0.85-2.5x (true P / n_lowres is
exactly 1 or 2). Every low-res -> full-res z mapping inherited the error.
How: tifffile writes ImageDescription '{"shape": [P, H, W]}' into IFD0, so one small HTTP range read per stack gives the
exact shape. Fallback (no description): walk the IFD chain counting pages (2 small reads per page).
Config: specimens_run, prefix (low-res cache), workers, meta_key; keys_key (optional R2 JSON key list: only these cached
        stacks); merge_base_key + merged_key (optional: write base table updated with this run's rows to merged_key;
        the base table itself is never overwritten).
Outputs: true_pages.jsonl, summary.json; R2 meta_key = {key: {pages, pages_est, n_lowres, ratio, compression, H, W, src}}.
"""
import io, json, struct, time, collections, concurrent.futures as cf
import numpy as np
import r2
import bigtiff_range as BT
from stages.lowres_cache import vial_code


def ifd_count(url, info):
    big = info["big"]; cw, ew, ow = (8, 20, 8) if big else (2, 12, 4)
    off, n = info["ifd0"], 0
    while off and n < 20000:
        cnt = struct.unpack("<Q" if big else "<H", BT.get(url, off, off + cw - 1))[0]
        off = struct.unpack("<Q" if big else "<I", BT.get(url, off + cw + cnt * ew, off + cw + cnt * ew + ow - 1))[0]
        n += 1
    return n


def main(cfg, out):
    import pandas as pd
    c = r2.client(); pre = cfg.get("prefix", "ant-twin/cache/lowres/")
    d = pd.read_csv(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")["Body"].read()))
    d = d[d.url_full.notna()]
    url = {f"{r['folder']}__{vial_code(r['vial'])}": r["url_full"] for r in d.to_dict("records")}
    keys = sorted(k[len(pre):-4] for k in r2.list_keys(c, r2.BUCKET, pre))
    if cfg.get("keys_key"):
        want = json.loads(c.get_object(Bucket=r2.BUCKET, Key=cfg["keys_key"])["Body"].read())
        print(f"key list {cfg['keys_key']}: {len(want)} keys, {len(set(want) - set(keys))} not in the low-res cache", flush=True)
        keys = sorted(set(keys) & set(want))
    print(f"cached stacks {len(keys)}, with url {sum(k in url for k in keys)}", flush=True)

    def one(key):
        t0 = time.time()
        try:
            z = np.load(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=pre + key + ".npz")["Body"].read()))
            nlow, pest = int(z["n_lowres"]), int(z["pages"])
            u = url[key]; info = BT.layout(u)
            big = info["big"]; t0_tags, _ = BT.parse_ifd(u, info["ifd0"], big)
            desc = t0_tags.get(270, (b"",))[0]
            P, src = None, None
            try:
                sh = json.loads(desc.rstrip(b"\x00").decode("utf-8", "replace"))["shape"]
                if len(sh) == 3 and sh[1] == info["H"] and sh[2] == info["W"]: P, src = int(sh[0]), "ifd0_shape"
            except Exception: pass
            if P is None:
                if info["compression"] == 1: P, src = pest, "uncompressed_stride"
                else: P, src = ifd_count(u, info), "ifd_walk"
            return {"key": key, "status": "ok", "pages": P, "pages_est": pest, "n_lowres": nlow, "ratio": round(P / nlow, 4),
                    "compression": info["compression"], "H": info["H"], "W": info["W"], "src": src, "est_ok": P == pest, "sec": round(time.time() - t0, 1)}
        except Exception as ex:
            return {"key": key, "status": "error", "error": repr(ex)[:200]}

    rows = []
    with cf.ThreadPoolExecutor(cfg.get("workers", 8)) as ex, open(out / "true_pages.jsonl", "w", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(one, keys)):
            rows.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 50 == 0 or rec["status"] != "ok": print(i, json.dumps(rec), flush=True)
    ok = [r for r in rows if r["status"] == "ok"]
    table = {r["key"]: {k: r[k] for k in ("pages", "pages_est", "n_lowres", "ratio", "compression", "H", "W", "src")} for r in ok}
    c.put_object(Bucket=r2.BUCKET, Key=cfg.get("meta_key", "ant-twin/meta/true_pages.json"), Body=json.dumps(table).encode())
    if cfg.get("merged_key"):
        base = json.loads(c.get_object(Bucket=r2.BUCKET, Key=cfg["merge_base_key"])["Body"].read())
        clash = sorted(k for k in table if k in base and base[k]["pages"] != table[k]["pages"])
        merged = {**base, **table}
        c.put_object(Bucket=r2.BUCKET, Key=cfg["merged_key"], Body=json.dumps(merged).encode())
        (out / "merge.json").write_text(json.dumps({"base": len(base), "new": len(table), "merged": len(merged), "pages_clash": clash}, indent=1))
        print(f"merged {len(base)} + {len(table)} -> {len(merged)} into {cfg['merged_key']}; page clashes {len(clash)}", flush=True)
    summ = {"n": len(rows), "status": dict(collections.Counter(r["status"] for r in rows)),
            "src": dict(collections.Counter(r["src"] for r in ok)),
            "compression": dict(collections.Counter(r["compression"] for r in ok)),
            "est_wrong": sum(not r["est_ok"] for r in ok),
            "est_wrong_by_compression": dict(collections.Counter(r["compression"] for r in ok if not r["est_ok"])),
            "ratio_values": dict(collections.Counter(round(r["ratio"], 2) for r in ok).most_common(12))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
