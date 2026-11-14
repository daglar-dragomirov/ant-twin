"""Stage seg_v1: brain segmentation adapted to AntScan without manual labels.

Dataset<id>_AntBrainMix = Toulkeridou 2023 stacks (OIST human labels, from the tk_volumes run) + AntScan brain boxes whose
draft was rated good by the automated gate-1 review (3 vision agents). AntScan boxes are resized in-plane so that
max(y, x) = target_px (the inference scale that transferred best) and repeated `dup` times. Species-level folds over both
sources. Trains nnU-Net 2d for one fold and reports overall and AntScan-only validation metrics.
Config: tk_run, boxes_run, nni_run, review_run, review_files, dataset_id, target_px, dup, n_folds, seed, fold, trainer,
        configuration, workdir.
Outputs: checkpoint_final.pth, plans.json, dataset.json, train_log/, val_per_case.json, summary.json, cases.json.
"""
import io, os, json, random, shutil, pathlib
import numpy as np
from scipy import ndimage as ndi
import r2
from stages.nnunet_tk import sh


def main(cfg, out):
    import nibabel as nib
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    did, conf, fold, tr = cfg["dataset_id"], cfg["configuration"], cfg["fold"], cfg["trainer"]
    name = f"Dataset{did}_AntBrainMix"; D = W / "raw" / name; (D / "imagesTr").mkdir(parents=True, exist_ok=True); (D / "labelsTr").mkdir(exist_ok=True)
    env = {**os.environ, "nnUNet_raw": str(W / "raw"), "nnUNet_preprocessed": str(W / "pre"), "nnUNet_results": str(W / "res"), "nnUNet_n_proc_DA": "16"}
    # 1) Toulkeridou cases
    tp = f"ant-twin/runs/{cfg['tk_run']}/outputs/"
    tk_fold = json.loads(get(tp + "species_folds.json")); case_sp = {}
    for k in r2.list_keys(c, r2.BUCKET, tp + "nnUNet_raw/"):
        sub, fn = k.split("/")[-2], k.split("/")[-1]
        if sub in ("imagesTr", "labelsTr"):
            p = D / sub / fn
            if not p.exists(): c.download_file(r2.BUCKET, k, str(p))
    for l in get(tp + "cases.jsonl").decode().splitlines():
        r = json.loads(l)
        if r.get("status") == "ok": case_sp[r["case"]] = ("TK", r["species"])
    # 2) AntScan good drafts
    bp, ap, rp = (f"ant-twin/runs/{cfg[k]}/outputs/" for k in ("boxes_run", "nni_run", "review_run"))
    man = {(m.get("key") or m["vial"]): m for m in (json.loads(l) for l in get(bp + "manifest.jsonl").decode().splitlines() if l.strip())}
    drafts = {json.loads(l)["key"]: json.loads(l) for l in get(ap + "drafts.jsonl").decode().splitlines() if l.strip()}
    rev = {}
    for k in cfg["review_files"]:
        for x in json.loads(get(k)): rev[x["key"]] = x
    good = {k: ("A" if x["verdict"] in ("A", "both") else "B") for k, x in rev.items() if x["verdict"] in ("A", "B", "both") and x.get("box_on_head") is not False}
    t = cfg["target_px"]; as_cases = []
    for i, (key, which) in enumerate(sorted(good.items())):
        m = man[key]; box = np.load(io.BytesIO(get(bp + f"boxes/{key}.npy")))
        if which == "A":
            a = np.load(io.BytesIO(get(ap + f"masks/{key}.npy"))); d = drafts[key]["downsample"]
            mk = np.zeros(box.shape, np.uint8); u = np.repeat(np.repeat(np.repeat(a, d, 0), d, 1), d, 2) if d > 1 else a
            mk[: min(u.shape[0], box.shape[0]), : min(u.shape[1], box.shape[1]), : min(u.shape[2], box.shape[2])] = u[: box.shape[0], : box.shape[1], : box.shape[2]]
        else:
            mk = np.load(io.BytesIO(get(rp + f"masks_B/{key}.npy")))
        s = t / max(box.shape[1], box.shape[2])
        img = ndi.zoom(box, (1, s, s), order=1); lab = (ndi.zoom(mk.astype(np.float32), (1, s, s), order=1) > 0.5).astype(np.uint8)
        for r_ in range(cfg.get("dup", 3)):
            case = f"AS_{i:03d}_{r_}"; case_sp[case] = ("AS", m["species"]); as_cases.append(case)
            nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), np.eye(4)), D / "imagesTr" / f"{case}_0000.nii.gz")
            nib.save(nib.Nifti1Image(np.ascontiguousarray(lab.transpose(2, 1, 0)), np.eye(4)), D / "labelsTr" / f"{case}.nii.gz")
    print(f"TK cases {sum(v[0] == 'TK' for v in case_sp.values())}, AntScan cases {len(as_cases)} from {len(good)} specimens", flush=True)
    (D / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "brain": 1}, "numTraining": len(case_sp),
                                                "file_ending": ".nii.gz", "description": "TK 2023 + auto-reviewed AntScan brains"}, indent=1))
    (out / "cases.json").write_text(json.dumps({k: list(v) for k, v in case_sp.items()}, indent=1))
    # 3) species-level folds: TK keeps its folds; AntScan species get random folds; duplicates stay together
    k = cfg["n_folds"]; as_sp = sorted({v[1] for v in case_sp.values() if v[0] == "AS"}); random.Random(cfg["seed"]).shuffle(as_sp)
    fo = {("TK", s): f for s, f in tk_fold.items()}; fo.update({("AS", s): j % k for j, s in enumerate(as_sp)})
    folds = [{"train": sorted(cs for cs, v in case_sp.items() if fo[tuple(v)] != fi), "val": sorted(cs for cs, v in case_sp.items() if fo[tuple(v)] == fi)} for fi in range(k)]
    if not (W / "pre" / name / "nnUNetPlans.json").exists():
        sh(["nnUNetv2_plan_and_preprocess", "-d", str(did), "-c", conf, "--verify_dataset_integrity", "-np", "16"], env)
    (W / "pre" / name / "splits_final.json").write_text(json.dumps(folds, indent=1)); shutil.copy(W / "pre" / name / "splits_final.json", out / "splits_final.json")
    # 4) train + 5) metrics, for each requested fold
    summ_all = {}
    for fold in cfg.get("folds", [fold]):
        fo_dir = out / f"fold_{fold}"; fo_dir.mkdir(exist_ok=True)
        sh(["nnUNetv2_train", str(did), conf, str(fold), "-tr", tr], env)
        R = W / "res" / name / f"{tr}__nnUNetPlans__{conf}"
        (fo_dir / "train_log").mkdir(exist_ok=True)
        for f in (R / f"fold_{fold}").glob("*"):
            if f.is_file() and f.suffix in (".txt", ".png", ".json"): shutil.copy(f, fo_dir / "train_log" / f.name)
        for f in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
            if (R / f).exists(): shutil.copy(R / f, out / f)
        shutil.copy(R / f"fold_{fold}" / "checkpoint_final.pth", fo_dir / "checkpoint_final.pth")
        V = R / f"fold_{fold}" / "validation"; per = {}
        for p_ in V.glob("*.nii.gz"):
            cs = p_.name[:-7]; ref = nib.load(str(D / "labelsTr" / p_.name)).get_fdata() > 0; pr = nib.load(str(p_)).get_fdata() > 0
            inter = int((ref & pr).sum()); per[cs] = {"source": case_sp[cs][0], "species": case_sp[cs][1], "dice": round(2 * inter / max(int(ref.sum() + pr.sum()), 1), 4),
                                                      "rel_vol_err": round((int(pr.sum()) - int(ref.sum())) / max(int(ref.sum()), 1), 4)}
        (fo_dir / "val_per_case.json").write_text(json.dumps(per, indent=1))
        summ = {}
        for src in ("TK", "AS"):
            v = [x for cs, x in per.items() if x["source"] == src and (src == "TK" or cs.endswith("_0"))]
            if v: summ[src] = {"n": len(v), "dice_median": float(np.median([x["dice"] for x in v])), "abs_vol_err_median": float(np.median([abs(x["rel_vol_err"]) for x in v]))}
        summ_all[str(fold)] = summ; print("fold", fold, "summary", summ, flush=True)
    (out / "summary.json").write_text(json.dumps(summ_all, indent=1)); print("summary", summ_all, flush=True)
