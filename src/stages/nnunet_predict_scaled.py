"""Stage nnunet_predict_scaled: apply a trained nnU-Net (2d) to AntScan brain boxes after in-plane rescaling.

Each box (z, y, x) is zoomed by s = target_px / max(y, x) along y and x only (2d model; z untouched), predicted, and the
mask is zoomed back with nearest neighbour. Several target sizes can be compared in one run.
Config: model_run (R2 run of nnunet_tk with checkpoint_final.pth, plans.json, dataset.json), dataset_id, configuration,
        fold, trainer, boxes_run, target_px (list), max_predict, workdir.
Outputs: pred_<t>/<key>.npy, qc_<t>/<key>.png, pred.jsonl (one line per key x target), summary.json.
"""
import os, io, json, shutil, pathlib, time
import numpy as np
from scipy import ndimage as ndi
import r2
from stages.nnunet_tk import sh, overlay


def stats(p):
    lab, n = ndi.label(p); sizes = ndi.sum(p, lab, range(1, n + 1)) if n else []
    edge = int(p[0].sum() + p[-1].sum() + p[:, 0].sum() + p[:, -1].sum() + p[:, :, 0].sum() + p[:, :, -1].sum())
    return {"mask_vox": int(p.sum()), "frac": round(float(p.mean()), 4), "n_components": int(n),
            "largest_comp_frac": round(float(max(sizes) / max(p.sum(), 1)), 4) if n else 0, "edge_vox": edge}


def main(cfg, out):
    import nibabel as nib
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client()
    env = {**os.environ, "nnUNet_raw": str(W / "raw"), "nnUNet_preprocessed": str(W / "pre"), "nnUNet_results": str(W / "res_scaled")}
    did, conf, fold, tr = cfg["dataset_id"], cfg["configuration"], cfg["fold"], cfg["trainer"]
    R = W / "res_scaled" / f"Dataset{did}_AntBrainTK" / f"{tr}__nnUNetPlans__{conf}"; (R / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    mp = f"ant-twin/runs/{cfg['model_run']}/outputs/"
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"): c.download_file(r2.BUCKET, mp + f, str(R / f))
    c.download_file(r2.BUCKET, mp + "checkpoint_final.pth", str(R / f"fold_{fold}" / "checkpoint_final.pth"))
    bp = f"ant-twin/runs/{cfg['boxes_run']}/outputs/"
    man = [json.loads(l) for l in c.get_object(Bucket=r2.BUCKET, Key=bp + "manifest.jsonl")["Body"].read().decode().splitlines() if l.strip()]
    ok = [m for m in man if m.get("status") == "ok"][: cfg.get("max_predict", 999)]
    boxes = {}
    for i, m in enumerate(ok):
        key = m.get("key") or m["vial"]
        boxes[f"AS_{i:03d}"] = (key, m, np.load(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=bp + f"boxes/{key}.npy")["Body"].read())))
    summ = {}
    with open(out / "pred.jsonl", "w", encoding="utf-8") as f:
        for t in cfg["target_px"]:
            I, O = W / f"in_{t}", W / f"out_{t}"
            shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); I.mkdir(parents=True); O.mkdir(parents=True)
            scale = {}
            for cid, (key, m, box) in boxes.items():
                s = t / max(box.shape[1], box.shape[2]); scale[cid] = s
                z = ndi.zoom(box, (1, s, s), order=1)
                nib.save(nib.Nifti1Image(np.ascontiguousarray(z.transpose(2, 1, 0)), np.eye(4)), I / f"{cid}_0000.nii.gz")
            sh(["nnUNetv2_predict", "-i", str(I), "-o", str(O), "-d", str(did), "-c", conf, "-f", str(fold), "-tr", tr], env)
            (out / f"pred_{t}").mkdir(exist_ok=True); (out / f"qc_{t}").mkdir(exist_ok=True); fr = []
            for cid, (key, m, box) in boxes.items():
                q = nib.load(str(O / f"{cid}.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0)
                p = ndi.zoom(q, [a / b for a, b in zip(box.shape, q.shape)], order=0)[: box.shape[0], : box.shape[1], : box.shape[2]]
                np.save(out / f"pred_{t}" / f"{key}.npy", p); overlay(box, p, out / f"qc_{t}" / f"{key}.png")
                rec = {"target_px": t, "key": key, "species": m["species"], "stained": m.get("stained"), "voxel_um": m.get("voxel_um"),
                       "box_shape": list(box.shape), "scale": round(scale[cid], 3), **stats(p)}
                fr.append(rec["largest_comp_frac"]); f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); print(json.dumps(rec, ensure_ascii=False), flush=True)
            summ[str(t)] = {"median_largest_comp_frac": float(np.median(fr)), "n_dominant_ge_0.9": int(sum(x >= 0.9 for x in fr)), "n": len(fr)}
            shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
