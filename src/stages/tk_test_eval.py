"""Stage tk_test_eval: independent accuracy of a TK nnU-Net model on the Toulkeridou 2023 testing.zip.

testing.zip holds the same kind of 2D slice stacks as training.zip but for species absent from training. Each stack
becomes a volume (as in tk_volumes), is predicted with the trained model, and is scored against its mask.
Config: r2_key, model_run, dataset_id, configuration, fold, trainer, workdir.
Outputs: per_stack.jsonl (Dice, volumes, relative volume error), per_species.json, summary.json, qc/<case>.png.
"""
import io, os, re, json, shutil, zipfile, pathlib, collections, time
import numpy as np, tifffile
import r2
from stages.nnunet_tk import sh
from stages.tk_volumes import species_of, qc

PAT = re.compile(r"^(?:[^/]+/)*(?!\._)(?P<stack>[^/]+)_(?P<idx>-?\d+)(?P<mask>_mask)?\.tif$")


def main(cfg, out):
    import nibabel as nib
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client()
    did, conf, fold, tr = cfg["dataset_id"], cfg["configuration"], cfg["fold"], cfg["trainer"]
    env = {**os.environ, "nnUNet_raw": str(W / "raw"), "nnUNet_preprocessed": str(W / "pre"), "nnUNet_results": str(W / "res_test")}
    R = W / "res_test" / f"Dataset{did}_AntBrainTK" / f"{tr}__nnUNetPlans__{conf}"; (R / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    mp = f"ant-twin/runs/{cfg['model_run']}/outputs/"
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"): c.download_file(r2.BUCKET, mp + f, str(R / f))
    c.download_file(r2.BUCKET, mp + "checkpoint_final.pth", str(R / f"fold_{fold}" / "checkpoint_final.pth"))
    zpath = pathlib.Path("/root/data/tk_testing.zip"); zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists(): t0 = time.time(); c.download_file(r2.BUCKET, cfg["r2_key"], str(zpath)); print(f"downloaded in {time.time()-t0:.0f}s", flush=True)
    zf = zipfile.ZipFile(zpath); stacks = collections.defaultdict(dict)
    for n in zf.namelist():
        if "__MACOSX" in n or "/._" in n: continue
        m = PAT.match(n)
        if m: stacks[m["stack"]].setdefault(int(m["idx"]), [None, None])[1 if m["mask"] else 0] = n
    print(f"{len(stacks)} stacks, {len({species_of(s) for s in stacks})} species", flush=True)
    I, O = W / "test_in", W / "test_out"
    shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); I.mkdir(parents=True); O.mkdir(parents=True)
    refs = {}
    for i, s in enumerate(sorted(stacks)):
        d = {k: v for k, v in stacks[s].items() if v[0] and v[1]}; idx = sorted(d)
        if len(idx) < 20: continue
        img = np.stack([tifffile.imread(io.BytesIO(zf.read(d[k][0]))) for k in idx])
        msk = (np.stack([tifffile.imread(io.BytesIO(zf.read(d[k][1]))) for k in idx]) > 127).astype(np.uint8)
        case = f"TT_{i:03d}"; refs[case] = (s, msk, img[len(img) // 2 - 1: len(img) // 2 + 2].copy())
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), np.eye(4)), I / f"{case}_0000.nii.gz")
    sh(["nnUNetv2_predict", "-i", str(I), "-o", str(O), "-d", str(did), "-c", conf, "-f", str(fold), "-tr", tr], env)
    (out / "qc").mkdir(exist_ok=True); recs = []
    with open(out / "per_stack.jsonl", "w", encoding="utf-8") as f:
        for case, (s, ref, _) in refs.items():
            p = nib.load(str(O / f"{case}.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0)
            inter = int((p & ref).sum()); vp, vr = int(p.sum()), int(ref.sum())
            r = {"case": case, "stack": s, "species": species_of(s), "dice": round(2 * inter / max(vp + vr, 1), 4),
                 "vol_pred": vp, "vol_ref": vr, "rel_vol_err": round((vp - vr) / max(vr, 1), 4)}
            recs.append(r); f.write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)
            if len(recs) <= 12:
                img = nib.load(str(I / f"{case}_0000.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0); qc(img, p, out / "qc" / f"{case}_{s}.png")
    sp = collections.defaultdict(list)
    for r in recs: sp[r["species"]].append(r)
    per_sp = {k: {"n_stacks": len(v), "dice_mean": round(float(np.mean([x["dice"] for x in v])), 4),
                  "rel_vol_err_mean": round(float(np.mean([x["rel_vol_err"] for x in v])), 4)} for k, v in sp.items()}
    (out / "per_species.json").write_text(json.dumps(per_sp, indent=1))
    ae = [abs(v["rel_vol_err_mean"]) for v in per_sp.values()]
    summ = {"n_stacks": len(recs), "n_species": len(per_sp), "dice_median": float(np.median([r["dice"] for r in recs])),
            "dice_mean": float(np.mean([r["dice"] for r in recs])), "species_abs_vol_err_median": float(np.median(ae)),
            "species_abs_vol_err_p90": float(np.percentile(ae, 90)), "species_vol_bias_mean": float(np.mean([v["rel_vol_err_mean"] for v in per_sp.values()]))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
    shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True)
