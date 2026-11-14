"""Stage seg_predict: brain segmentation and brain volume for every cached boxes_v3 box.

Model: seg v1 nnU-Net 2d (Toulkeridou 2023 + auto-reviewed AntScan), ensemble of the folds listed in cfg (each fold's
checkpoint from its R2 run). Each box is resized in-plane to target_px (the training scale), predicted in batches,
mapped back with nearest neighbour; the largest 3D component (after a 1-voxel opening) is the brain.
Config: models (list of {fold, run, path}), plans_run, dataset_id, configuration, trainer, target_px, batch, prefix,
        max_specimens, workdir.
Outputs: brains.jsonl (key, brain_vox, brain_um3, components, edge contact, fraction), qc/<key>.jpg (8 slices, red
         contour), masks in R2 ant-twin/cache/brain_masks_v1/<key>.npz, summary.json.
"""
import io, os, json, shutil, pathlib, time
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import r2
from stages.nnunet_tk import sh


def largest(m):
    m = ndi.binary_opening(m > 0, iterations=1); lab, n = ndi.label(m)
    if n == 0: return m.astype(np.uint8), 0, 0.0
    s = ndi.sum(m, lab, range(1, n + 1)); k = int(np.argmax(s))
    return (lab == k + 1).astype(np.uint8), int(n), float(s[k] / s.sum())


def qc_jpg(box, m, path):
    zz = np.nonzero(m.any((1, 2)))[0]
    ks = [int(v) for v in np.linspace(zz.min(), zz.max(), 10)[1:-1]] if len(zz) else [int(x) for x in np.linspace(0, box.shape[0] - 1, 8)]
    lo, hi = np.percentile(box, [1, 99.8]); t = []
    for k in ks:
        a = np.clip((box[k].astype(np.float32) - lo) / max(hi - lo, 1), 0, 1); rgb = np.stack([a] * 3, -1)
        mk = m[k] > 0; e = mk & ~ndi.binary_erosion(mk); rgb[e] = [1, .15, .15]; t.append((rgb * 255).astype(np.uint8))
    M = np.concatenate([np.concatenate(t[:4], 1), np.concatenate(t[4:], 1)], 0)
    im = Image.fromarray(M); im.thumbnail((1600, 1600)); im.save(path, quality=82)


def main(cfg, out):
    import nibabel as nib
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    did, conf, tr = cfg["dataset_id"], cfg["configuration"], cfg["trainer"]
    R = W / "res_seg" / f"Dataset{did}_AntBrainMix" / f"{tr}__nnUNetPlans__{conf}"; R.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "nnUNet_raw": str(W / "raw"), "nnUNet_preprocessed": str(W / "pre"), "nnUNet_results": str(W / "res_seg")}
    (W / "raw").mkdir(parents=True, exist_ok=True)
    pp = f"ant-twin/runs/{cfg['plans_run']}/outputs/"
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"): c.download_file(r2.BUCKET, pp + f, str(R / f))
    folds = []
    for m in cfg["models"]:
        (R / f"fold_{m['fold']}").mkdir(exist_ok=True)
        try: c.download_file(r2.BUCKET, f"ant-twin/runs/{m['run']}/outputs/{m['path']}", str(R / f"fold_{m['fold']}" / "checkpoint_final.pth")); folds.append(str(m["fold"]))
        except Exception as ex: print("model not available", m, ex, flush=True)
    print("ensemble folds:", folds, flush=True)
    pre = cfg.get("prefix", "ant-twin/cache/boxes_v3/"); mpre = cfg.get("mask_prefix", "ant-twin/cache/brain_masks_v1/")
    keys = sorted(k[len(pre):-4] for k in r2.list_keys(c, r2.BUCKET, pre))[: cfg.get("max_specimens", 100000)]
    done = {k[len(mpre):-4] for k in r2.list_keys(c, r2.BUCKET, mpre)}
    todo = [k for k in keys if k not in done]; print(f"boxes {len(keys)}, already segmented {len(done)}, to do {len(todo)}", flush=True)
    (out / "qc").mkdir(exist_ok=True); t = cfg["target_px"]; B = cfg.get("batch", 12)
    with open(out / "brains.jsonl", "w", encoding="utf-8") as fo:
        for b0 in range(0, len(todo), B):
            t0 = time.time(); I, O = W / "seg_in", W / "seg_out"
            shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); I.mkdir(parents=True); O.mkdir(parents=True)
            items = {}
            for j, key in enumerate(todo[b0: b0 + B]):
                z = np.load(io.BytesIO(get(pre + key + ".npz"))); box = z["box"]; vox = float(z["voxel_um"])
                s = t / max(box.shape[1], box.shape[2]); img = ndi.zoom(box, (1, s, s), order=1)
                cid = f"S_{b0 + j:04d}"; items[cid] = (key, box, vox, z["box_zyx"])
                nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), np.eye(4)), I / f"{cid}_0000.nii.gz")
            sh(["nnUNetv2_predict", "-i", str(I), "-o", str(O), "-d", str(did), "-c", conf, "-f", *folds, "-tr", tr, "--disable_progress_bar"], env)
            for cid, (key, box, vox, bz) in items.items():
                q = nib.load(str(O / f"{cid}.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0)
                p = ndi.zoom(q, [a / b for a, b in zip(box.shape, q.shape)], order=0)[: box.shape[0], : box.shape[1], : box.shape[2]]
                m, ncomp, lfrac = largest(p)
                edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m[:, :, 0].any() or m[:, :, -1].any())
                buf = io.BytesIO(); np.savez_compressed(buf, mask=m, box_zyx=bz, voxel_um=vox); c.put_object(Bucket=r2.BUCKET, Key=mpre + key + ".npz", Body=buf.getvalue())
                qc_jpg(box, m, out / "qc" / f"{key}.jpg")
                rec = {"key": key, "voxel_um": vox, "box_shape": list(box.shape), "brain_vox": int(m.sum()), "brain_um3": float(m.sum() * vox ** 3),
                       "n_components": ncomp, "largest_frac": round(lfrac, 4), "touches_edge": edge, "box_frac": round(float(m.mean()), 4), "folds": folds}
                fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"batch {b0}-{b0 + len(items)} in {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(W / "seg_in", ignore_errors=True); shutil.rmtree(W / "seg_out", ignore_errors=True)
    rows = [json.loads(l) for l in open(out / "brains.jsonl", encoding="utf-8")]
    summ = {"n": len(rows), "empty": sum(r["brain_vox"] == 0 for r in rows), "touch_edge": sum(r["touches_edge"] for r in rows), "folds": folds}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
