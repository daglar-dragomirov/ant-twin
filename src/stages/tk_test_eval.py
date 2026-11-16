"""Stage tk_test_eval: independent accuracy of a TK nnU-Net model on the Toulkeridou 2023 testing.zip.

testing.zip holds the same kind of 2D slice stacks as training.zip but for species absent from training. Each stack
becomes a volume (as in tk_volumes), is predicted with the trained model, and is scored against its mask.
Config: r2_key, train_dataset_run (tk_volumes run, for the training-species list), model_run, dataset_id, configuration, fold, trainer, workdir.
Outputs: per_stack.jsonl (Dice, volumes, relative volume error), per_species.json, summary.json, qc/<case>.png.
"""
import io, os, re, json, shutil, zipfile, pathlib, collections, time
import numpy as np, tifffile
import r2
from stages.nnunet_tk import sh
from stages.tk_volumes import species_of, qc

ALIAS = {"Gamp_sp": "Gnamp_sp"}  # typo in the Dryad mask file names


def parse_name(b):
    """Two naming schemes in testing.zip: <stack>_<i>.tif + <stack>_<i>_mask.tif, and <stack><i>.tif + <stack>[_]_mask<i>.tif."""
    for rx, is_mask in ((r"^(.+)_(\d+)_mask$", True), (r"^(.+?)_*mask(\d+)$", True), (r"^(.+?)_(\d+)$", False), (r"^(.+?)(\d+)$", False)):
        m = re.match(rx, b)
        if m: return m.group(1).rstrip("_"), int(m.group(2)), is_mask
    return None


def species_norm(x):
    return re.sub(r"[^a-z]", "", x.lower())


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
        if "__MACOSX" in n or "/._" in n or not n.endswith(".tif"): continue
        b = n.rsplit("/", 1)[-1][:-4]
        st = parse_name(b)
        if st: stacks[ALIAS.get(st[0], st[0])].setdefault(st[1], [None, None])[1 if st[2] else 0] = n
    train_sp = set(json.loads(c.get_object(Bucket=r2.BUCKET, Key=f"ant-twin/runs/{cfg['train_dataset_run']}/outputs/species_folds.json")["Body"].read()))
    print(f"{len(stacks)} stacks, {len({species_of(s) for s in stacks})} species; training species: {len(train_sp)}", flush=True)
    I, O = W / "test_in", W / "test_out"
    shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); I.mkdir(parents=True); O.mkdir(parents=True)
    refs = {}; meta = {}
    for i, s in enumerate(sorted(stacks)):
        imgs = {k: v[0] for k, v in stacks[s].items() if v[0]}; msks = {k: v[1] for k, v in stacks[s].items() if v[1]}
        both = sorted(set(imgs) & set(msks))
        if len(both) >= 0.9 * max(len(imgs), len(msks)): pairs, mode = [(imgs[k], msks[k]) for k in both], "by_index"
        elif len(imgs) == len(msks): pairs, mode = list(zip([imgs[k] for k in sorted(imgs)], [msks[k] for k in sorted(msks)])), "by_order"
        else: print("unpaired", s, len(imgs), len(msks), len(both), flush=True); continue
        if len(pairs) < 20: continue
        img = np.stack([tifffile.imread(io.BytesIO(zf.read(a))) for a, _ in pairs])
        raw = np.stack([tifffile.imread(io.BytesIO(zf.read(b))) for _, b in pairs])
        k = len(raw) // 2; border = np.concatenate([raw[k, 0], raw[k, -1], raw[k, :, 0], raw[k, :, -1]])
        inverted = bool((border > 127).mean() > 0.5)  # background touches the crop border; if it is white, labels are inverted
        msk = ((raw <= 127) if inverted else (raw > 127)).astype(np.uint8)
        case = f"TT_{i:03d}"; refs[case] = (s, msk, None)
        meta[case] = {"pairing": mode, "inverted_mask": inverted, "in_training": species_norm(species_of(s)) in {species_norm(x) for x in train_sp}}
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), np.eye(4)), I / f"{case}_0000.nii.gz")
    sh(["nnUNetv2_predict", "-i", str(I), "-o", str(O), "-d", str(did), "-c", conf, "-f", str(fold), "-tr", tr], env)
    (out / "qc").mkdir(exist_ok=True); recs = []
    with open(out / "per_stack.jsonl", "w", encoding="utf-8") as f:
        for case, (s, ref, _) in refs.items():
            p = nib.load(str(O / f"{case}.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0)
            inter = int((p & ref).sum()); vp, vr = int(p.sum()), int(ref.sum())
            r = {"case": case, "stack": s, "species": species_of(s), "dice": round(2 * inter / max(vp + vr, 1), 4),
                 "vol_pred": vp, "vol_ref": vr, "rel_vol_err": round((vp - vr) / max(vr, 1), 4), **meta[case]}
            recs.append(r); f.write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True)
            if len(recs) <= 12:
                img = nib.load(str(I / f"{case}_0000.nii.gz")).get_fdata().astype(np.uint8).transpose(2, 1, 0); qc(img, p, out / "qc" / f"{case}_{s}.png")
    sp = collections.defaultdict(list)
    for r in recs:
        if not r["in_training"]: sp[r["species"]].append(r)  # independent metric: unseen species only
    per_sp = {k: {"n_stacks": len(v), "dice_mean": round(float(np.mean([x["dice"] for x in v])), 4),
                  "rel_vol_err_mean": round(float(np.mean([x["rel_vol_err"] for x in v])), 4)} for k, v in sp.items()}
    (out / "per_species.json").write_text(json.dumps(per_sp, indent=1))
    ae = [abs(v["rel_vol_err_mean"]) for v in per_sp.values()]
    ind = [r for r in recs if not r["in_training"]]
    summ = {"n_stacks_all": len(recs), "n_inverted": sum(r["inverted_mask"] for r in recs), "n_in_training": len(recs) - len(ind),
            "dice_median_all": float(np.median([r["dice"] for r in recs])), "n_stacks": len(ind), "n_species": len(per_sp), "dice_median": float(np.median([r["dice"] for r in ind])),
            "dice_mean": float(np.mean([r["dice"] for r in ind])), "species_abs_vol_err_median": float(np.median(ae)),
            "species_abs_vol_err_p90": float(np.percentile(ae, 90)), "species_vol_bias_mean": float(np.mean([v["rel_vol_err_mean"] for v in per_sp.values()]))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
    shutil.rmtree(I, ignore_errors=True); shutil.rmtree(O, ignore_errors=True)
