"""Stage finder_predict: locate the brain on whole-specimen AntScan low-res stacks with the trained low-res finder,
and turn each detection into a full-resolution brain box.

Streams specimens in batches (fetch low-res stack -> downsample -> nnU-Net predict -> delete) to fit a small disk.
Config: specimens_run, model_run, dataset_id, dataset_name, configuration, fold, trainer, ds, batch, margin_frac,
        workers, max_specimens, qc_every, workdir, only_workers_with_colony.
Outputs: finder.jsonl (one line per specimen: status, low-res bbox, full-res box_zyx, low-res brain volume um3,
         component stats), qc/<key>.png for every qc_every-th specimen, summary.json.
"""
import io, os, json, shutil, pathlib, time, concurrent.futures as cf
import numpy as np
from scipy import ndimage as ndi
import r2
import bigtiff_range as BT
from stages.nnunet_tk import sh
from stages.tk_volumes import qc
from stages.lowres_finder_data import lowres_stack


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def main(cfg, out):
    import nibabel as nib, pandas as pd
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client()
    did, conf, fold, tr, name = cfg["dataset_id"], cfg["configuration"], cfg["fold"], cfg["trainer"], f"Dataset{cfg['dataset_id']}_{cfg['dataset_name']}"
    env = {**os.environ, "nnUNet_raw": str(W / "raw"), "nnUNet_preprocessed": str(W / "pre"), "nnUNet_results": str(W / "res_finder")}
    R = W / "res_finder" / name / f"{tr}__nnUNetPlans__{conf}"; (R / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    mp = f"ant-twin/runs/{cfg['model_run']}/outputs/"
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"): c.download_file(r2.BUCKET, mp + f, str(R / f))
    c.download_file(r2.BUCKET, mp + "checkpoint_final.pth", str(R / f"fold_{fold}" / "checkpoint_final.pth"))
    d = pd.read_csv(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")["Body"].read()))
    d = d[(d.caste == "worker") & d.url_full.notna() & d.url_lowres_slice0.notna()]
    if cfg.get("only_workers_with_colony", True): d = d[d.colony_any.notna()]
    recs = d.to_dict("records")[: cfg.get("max_specimens", 100000)]
    si, sn = cfg.get("shard", [0, 1]); recs = recs[si::sn]  # shard i of n
    print(f"specimens: {len(recs)}", flush=True)
    ds, B = int(cfg.get("ds", 2)), int(cfg.get("batch", 16)); (out / "qc").mkdir(exist_ok=True)

    def prep(i_r):
        i, r = i_r; key = f"{r['folder']}__{vial_code(r['vial'])}"
        try:
            try:  # R2 low-res cache first (lowres_cache node); the KIT server only as a fallback
                z = np.load(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=cfg.get("cache_prefix", "ant-twin/cache/lowres/") + key + ".npz")["Body"].read()))
                v, nlow, P, H_, W_ = z["img"], int(z["n_lowres"]), int(z["pages"]), int(z["H"]), int(z["W"])
            except Exception:
                v, nlow = lowres_stack(r["url_lowres_slice0"].replace("slice_0000.png", "slice_%04d.png"))
                if v is None: return key, r, None, {"status": "no_lowres"}
                info = BT.layout(r["url_full"]); size, P = BT.n_pages(r["url_full"], info); H_, W_ = info["H"], info["W"]
            vox = float(r["voxel_um"]); fz, fy, fx = P / nlow, H_ / v.shape[1], W_ / v.shape[2]
            sp = (vox * fz * ds, vox * fy * ds, vox * fx * ds); img = v[::ds, ::ds, ::ds]
            geo = {"P": P, "H": H_, "W": W_, "nlow": nlow, "lowres_shape": list(v.shape), "f": [fz, fy, fx], "spacing_um": sp}
            return key, r, img, geo
        except Exception as ex:
            return key, r, None, {"status": "error", "error": repr(ex)[:200]}

    def done_elsewhere():
        ks = set()
        for run in cfg.get("skip_runs", []):
            try:
                for l in c.get_object(Bucket=r2.BUCKET, Key=f"ant-twin/runs/{run}/outputs/finder.jsonl")["Body"].read().decode().splitlines():
                    if l.strip(): ks.add(json.loads(l)["key"])
            except Exception: pass
        return ks
    keyof = lambda r: f"{r['folder']}__{vial_code(r['vial'])}"
    skip = done_elsewhere(); recs = [r for r in recs if keyof(r) not in skip]; print(f"after skipping done keys: {len(recs)}", flush=True)
    n_done = 0
    with open(out / "finder.jsonl", "w", encoding="utf-8") as fo:
        for b0 in range(0, len(recs), B):
            skip = done_elsewhere()
            if all(keyof(r) in skip for r in recs[b0: b0 + B]): continue
            t0 = time.time(); I, O = W / "fin_in", W / "fin_out"
            shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); I.mkdir(parents=True); O.mkdir(parents=True)
            batch = [(i, r) for i, r in enumerate(recs[b0: b0 + B], b0) if keyof(r) not in skip]
            with cf.ThreadPoolExecutor(cfg.get("workers", 6)) as ex: got = list(ex.map(prep, batch))
            todo = {}
            for j, (key, r, img, geo) in enumerate(got):
                if img is None:
                    fo.write(json.dumps({"key": key, "species": r["name"], **geo}) + "\n"); continue
                cid = f"F_{b0 + j:04d}"; todo[cid] = (key, r, img, geo); sp = geo["spacing_um"]
                nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), np.diag([sp[2], sp[1], sp[0], 1.0])), I / f"{cid}_0000.nii.gz")
            if todo: sh(["nnUNetv2_predict", "-i", str(I), "-o", str(O), "-d", str(did), "-c", conf, "-f", str(fold), "-tr", tr, "--disable_progress_bar", "-npp", str(cfg.get("npp", 1)), "-nps", str(cfg.get("nps", 1))] + (["--disable_tta"] if cfg.get("disable_tta", True) else []), env)
            for cid, (key, r, img, geo) in todo.items():
                p = nib.load(str(O / f"{cid}.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0)
                lab, n = ndi.label(p); rec = {"key": key, "species": r["name"], "species_key": r["species_key"], "subfamily": r["subfamily"],
                                               "stained": r["stained"], "voxel_um": r["voxel_um"], "folder": r["folder"], "vial": r["vial"],
                                               "url_full": r["url_full"], "n_components": int(n), **{k: geo[k] for k in ("P", "H", "W", "nlow", "lowres_shape", "spacing_um")}}
                if n == 0: rec["status"] = "no_brain"; fo.write(json.dumps(rec) + "\n"); continue
                sizes = ndi.sum(p, lab, range(1, n + 1)); big = lab == (int(np.argmax(sizes)) + 1)
                cand = []  # top-3 components (fallback candidates for a second round), low-res bbox + size
                for ci in np.argsort(sizes)[::-1][:3]:
                    cz, cy, cx = np.nonzero(lab == ci + 1)
                    cand.append({"vox": int(sizes[ci]), "lowres_bbox": [int(cz.min() * ds), int(cy.min() * ds), int(cx.min() * ds), int((cz.max() + 1) * ds), int((cy.max() + 1) * ds), int((cx.max() + 1) * ds)]})
                rec["candidates"] = cand
                zz, yy, xx = np.nonzero(big); sp = geo["spacing_um"]
                lo = np.array([zz.min(), yy.min(), xx.min()]) * ds; hi = (np.array([zz.max(), yy.max(), xx.max()]) + 1) * ds
                f = np.array(geo["f"]); c_ = (lo + hi) / 2 * f; half = (hi - lo) / 2 * f * (1 + cfg.get("margin_frac", 0.4))
                lim = np.array([geo["P"], geo["H"], geo["W"]])
                z0, y0, x0 = np.maximum(np.floor(c_ - half), 0).astype(int); z1, y1, x1 = np.minimum(np.ceil(c_ + half), lim).astype(int)
                rec.update(status="ok", lowres_bbox=[int(x) for x in (*lo, *hi)], box_zyx=[int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                           brain_vol_lowres_um3=float(big.sum() * sp[0] * sp[1] * sp[2]), largest_comp_frac=round(float(sizes.max() / sizes.sum()), 4))
                fo.write(json.dumps(rec) + "\n")
                if n_done % cfg.get("qc_every", 10) == 0:
                    k = int(np.median(zz)); sl = slice(max(0, k - 30), min(img.shape[0], k + 30)); qc(img[sl], big[sl].astype(np.uint8), out / "qc" / f"{key}.png")
                n_done += 1
            fo.flush(); print(f"batch {b0}-{b0 + len(got)} done in {time.time() - t0:.0f}s; ok so far {n_done}", flush=True)
    shutil.rmtree(W / "fin_in", ignore_errors=True); shutil.rmtree(W / "fin_out", ignore_errors=True)
    rows = [json.loads(l) for l in open(out / "finder.jsonl", encoding="utf-8")]
    from collections import Counter
    summ = {"n": len(rows), "status": dict(Counter(r.get("status") for r in rows))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
