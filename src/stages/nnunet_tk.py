"""Stage nnunet_tk: train nnU-Net v2 on the Toulkeridou 2023 dataset built by tk_volumes, then predict AntScan boxes.

Config: dataset_run (R2 run of tk_volumes), dataset_id, configuration ("2d"/"3d_fullres"), fold, trainer,
        boxes_run (R2 run of brain_boxes v2; optional), max_predict, workdir (big scratch outside outputs/).
Outputs: train_log/ (nnU-Net fold log, progress.png, validation summary), checkpoint_final.pth, plans,
         pred/<key>.npy (uint8 mask on the box grid), pred_qc/<key>.png, pred.jsonl, summary.json.
Heavy preprocessed data stays in workdir (not synced); everything needed to reuse the model is in outputs/.
"""
import os, io, json, shutil, subprocess, pathlib, time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import r2


def sh(cmd, env):
    print("$", " ".join(cmd), flush=True); t0 = time.time()
    subprocess.run(cmd, env=env, check=True); print(f"  done in {time.time()-t0:.0f}s", flush=True)


def overlay(vol, m, path):
    zz = np.nonzero(m.any((1, 2)))[0]
    ks = [int(v) for v in np.linspace(zz.min(), zz.max(), 8)] if len(zz) else [vol.shape[0] // 2] * 8
    lo, hi = np.percentile(vol, [1, 99.8]); t = []
    for k in ks:
        a = np.clip((vol[k].astype(np.float32) - lo) / max(hi - lo, 1), 0, 1); rgb = np.stack([a] * 3, -1)
        mk = m[k] > 0; e = mk & ~ndi.binary_erosion(mk); rgb[e] = [1, 0.1, 0.1]; t.append((rgb * 255).astype(np.uint8))
    M = np.concatenate([np.concatenate(t[:4], 1), np.concatenate(t[4:], 1)], 0)
    im = Image.fromarray(M); im.thumbnail((2000, 2000)); im.save(path)


def main(cfg, out):
    import nibabel as nib
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client()
    env = {**os.environ, "nnUNet_raw": str(W / "raw"), "nnUNet_preprocessed": str(W / "pre"), "nnUNet_results": str(W / "res"),
           "nnUNet_n_proc_DA": str(cfg.get("n_proc_da", 8))}
    did, conf, fold, tr = cfg["dataset_id"], cfg["configuration"], cfg["fold"], cfg["trainer"]
    name = f"Dataset{did}_AntBrainTK"
    # 1) fetch dataset from R2
    pre = f"ant-twin/runs/{cfg['dataset_run']}/outputs/"
    keys = r2.list_keys(c, r2.BUCKET, pre + f"nnUNet_raw/{name}/")
    for k in keys:
        p = W / "raw" / k.split("nnUNet_raw/")[1]
        if not p.exists(): p.parent.mkdir(parents=True, exist_ok=True); c.download_file(r2.BUCKET, k, str(p))
    print(f"dataset files: {len(keys)}", flush=True)
    # 2) plan + preprocess, then install the species-level splits
    if not (W / "pre" / name / "nnUNetPlans.json").exists():
        sh(["nnUNetv2_plan_and_preprocess", "-d", str(did), "-c", conf, "--verify_dataset_integrity", "-np", str(cfg.get("np", 8))], env)
    c.download_file(r2.BUCKET, pre + "splits_final.json", str(W / "pre" / name / "splits_final.json"))
    # 3) train
    sh(["nnUNetv2_train", str(did), conf, str(fold), "-tr", tr], env)
    R = W / "res" / name / f"{tr}__nnUNetPlans__{conf}"
    (out / "train_log").mkdir(exist_ok=True)
    for f in (R / f"fold_{fold}").glob("*"):
        if f.is_file() and f.suffix in (".txt", ".png", ".json"): shutil.copy(f, out / "train_log" / f.name)
    for f in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        if (R / f).exists(): shutil.copy(R / f, out / f)
    shutil.copy(R / f"fold_{fold}" / "checkpoint_final.pth", out / "checkpoint_final.pth")
    summ = {}
    vs = R / f"fold_{fold}" / "validation" / "summary.json"
    if vs.exists():
        s = json.loads(vs.read_text()); summ["val_dice_brain"] = s.get("foreground_mean", {}).get("Dice"); shutil.copy(vs, out / "val_summary.json")
    print("validation", summ, flush=True)
    # 4) transfer: predict AntScan boxes
    if cfg.get("boxes_run"):
        bp = f"ant-twin/runs/{cfg['boxes_run']}/outputs/"
        man = [json.loads(l) for l in c.get_object(Bucket=r2.BUCKET, Key=bp + "manifest.jsonl")["Body"].read().decode().splitlines() if l.strip()]
        ok = [m for m in man if m.get("status") == "ok"][: cfg.get("max_predict", 999)]
        I, O = W / "pred_in", W / "pred_out"
        shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); I.mkdir(parents=True); O.mkdir(parents=True)
        boxes = {}
        for i, m in enumerate(ok):
            key = m.get("key") or m["vial"]
            box = np.load(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=bp + f"boxes/{key}.npy")["Body"].read()))
            ds = max(1, round(cfg.get("target_um", 2.44) / float(m.get("voxel_um", 2.44))))
            box = box[::ds, ::ds, ::ds] if ds > 1 else box; boxes[f"AS_{i:03d}"] = (key, m, box)
            nib.save(nib.Nifti1Image(np.ascontiguousarray(box.transpose(2, 1, 0)), np.eye(4)), I / f"AS_{i:03d}_0000.nii.gz")
        sh(["nnUNetv2_predict", "-i", str(I), "-o", str(O), "-d", str(did), "-c", conf, "-f", str(fold), "-tr", tr], env)
        (out / "pred").mkdir(exist_ok=True); (out / "pred_qc").mkdir(exist_ok=True)
        with open(out / "pred.jsonl", "w", encoding="utf-8") as f:
            for cid, (key, m, box) in boxes.items():
                p = nib.load(str(O / f"{cid}.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0)
                lab, n = ndi.label(p); sizes = ndi.sum(p, lab, range(1, n + 1)) if n else []
                np.save(out / "pred" / f"{key}.npy", p); overlay(box, p, out / "pred_qc" / f"{key}.png")
                edge = int(p[0].sum() + p[-1].sum() + p[:, 0].sum() + p[:, -1].sum() + p[:, :, 0].sum() + p[:, :, -1].sum())
                rec = {"key": key, "species": m["species"], "stained": m.get("stained"), "folder": m.get("folder"), "voxel_um": m.get("voxel_um"),
                       "box_shape": list(box.shape), "mask_vox": int(p.sum()), "frac": round(float(p.mean()), 4), "n_components": int(n),
                       "largest_comp_frac": round(float(max(sizes) / max(p.sum(), 1)), 4) if n else 0, "edge_vox": edge}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); print(json.dumps(rec, ensure_ascii=False), flush=True)
        summ["n_predicted"] = len(boxes)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
