"""Stage tk_volumes: reuse the OIST brain labels of Toulkeridou et al. 2023 (Dryad 10.5061/dryad.qz612jmgv, CC0).

training.zip holds 2D slices <stack>_<idx>.tif + <stack>_<idx>_mask.tif (520x520 uint8; masks are soft 0..255).
Stacks <sp>, <sp>2, <sp>3 are the same species re-sliced along different axes, each axis cropped and stretched to
520x520 separately (a native-grid 3D fusion was tested on 2 species and rejected: axis-mask Dice 0.38-0.72). So every
stack becomes one labelled volume (N, 520, 520) with unit spacing, and cross-validation folds are split BY SPECIES so
re-slices of one head never sit in both train and validation (leakage guard).
Config: r2_key (zip in R2), local_zip (optional), dataset_id, n_folds, seed, only (optional species list).
Outputs: cases.jsonl, qc/<case>.png, nnUNet_raw/Dataset<id>_AntBrainTK/{imagesTr,labelsTr,dataset.json}, splits_final.json.
"""
import re, io, json, zipfile, pathlib, collections, random, time, tempfile
import numpy as np, tifffile
from scipy import ndimage as ndi
from PIL import Image
import r2

PAT = re.compile(r"^training/(?P<stack>[^/]+)_(?P<idx>-?\d+)(?P<mask>_mask)?\.tif$")


def species_of(stack):
    m = re.match(r"^(.*?)([23])$", stack)
    return (m.group(1) if m else stack).replace("-", "_").rstrip("_")


def qc(img, m, path):
    tiles = []
    for ax in range(3):
        k = img.shape[ax] // 2
        im = np.take(img, k, ax); mk = np.take(m, k, ax) > 0
        e = mk & ~ndi.binary_erosion(mk, iterations=1)
        rgb = np.stack([im] * 3, -1).copy(); rgb[e] = [255, 40, 40]; tiles.append(rgb)
    h = max(t.shape[0] for t in tiles)
    row = np.concatenate([np.pad(t, ((0, h - t.shape[0]), (0, 4), (0, 0))) for t in tiles], 1)
    im = Image.fromarray(row); im.thumbnail((1500, 600)); im.save(path)


def main(cfg, out):
    import nibabel as nib
    zpath = cfg.get("local_zip")
    if not zpath or not pathlib.Path(zpath).exists():
        zpath = str(pathlib.Path(tempfile.gettempdir()) / "tk_training.zip")
        if not pathlib.Path(zpath).exists():
            t0 = time.time(); r2.client().download_file(r2.BUCKET, cfg["r2_key"], zpath); print(f"downloaded zip in {time.time()-t0:.0f}s", flush=True)
    zf = zipfile.ZipFile(zpath)
    stacks = collections.defaultdict(dict)
    for n in zf.namelist():
        m = PAT.match(n)
        if m: stacks[m["stack"]].setdefault(int(m["idx"]), [None, None])[1 if m["mask"] else 0] = n
    sp_of = {s: species_of(s) for s in stacks}
    print(f"{len(stacks)} stacks, {len(set(sp_of.values()))} species", flush=True)
    D = out / "nnUNet_raw" / f"Dataset{cfg['dataset_id']}_AntBrainTK"; (D / "imagesTr").mkdir(parents=True, exist_ok=True); (D / "labelsTr").mkdir(exist_ok=True)
    (out / "qc").mkdir(exist_ok=True)
    cases = {}
    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for i, s in enumerate(sorted(stacks)):
            if cfg.get("only") and sp_of[s] not in cfg["only"]: continue
            t0 = time.time(); d = {k: v for k, v in stacks[s].items() if v[0] and v[1]}; idx = sorted(d)
            img = np.stack([tifffile.imread(io.BytesIO(zf.read(d[k][0]))) for k in idx])
            msk = np.stack([tifffile.imread(io.BytesIO(zf.read(d[k][1]))) for k in idx]) > 127
            case = f"TK_{i:03d}"
            rec = {"case": case, "stack": s, "species": sp_of[s], "n_slices": len(idx), "idx_range": [idx[0], idx[-1]],
                   "gaps": int(idx[-1] - idx[0] + 1 - len(idx)), "unpaired": len(stacks[s]) - len(d),
                   "shape": list(img.shape), "brain_vox": int(msk.sum()), "brain_frac": round(float(msk.mean()), 4),
                   "slices_with_brain": int(msk.any((1, 2)).sum())}
            rec["status"] = "ok" if rec["brain_vox"] > 0 and rec["n_slices"] >= 20 else "skip"
            if rec["status"] == "ok":
                aff = np.eye(4)  # unit spacing: in-plane stretch to 520x520 is unknown per stack (documented)
                nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), aff), D / "imagesTr" / f"{case}_0000.nii.gz")
                nib.save(nib.Nifti1Image(np.ascontiguousarray(msk.astype(np.uint8).transpose(2, 1, 0)), aff), D / "labelsTr" / f"{case}.nii.gz")
                qc(img, msk, out / "qc" / f"{case}_{s}.png"); cases[case] = sp_of[s]
            rec["sec"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec), flush=True)
    (D / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "brain": 1},
                                                "numTraining": len(cases), "file_ending": ".nii.gz",
                                                "description": "Toulkeridou et al. 2023 (Dryad qz612jmgv, CC0) brain labels; one case per re-sliced stack"}, indent=1))
    sps = sorted(set(cases.values())); random.Random(cfg["seed"]).shuffle(sps); k = cfg["n_folds"]
    fold_of = {sp: j % k for j, sp in enumerate(sps)}
    folds = [{"train": sorted(c for c, sp in cases.items() if fold_of[sp] != fi), "val": sorted(c for c, sp in cases.items() if fold_of[sp] == fi)} for fi in range(k)]
    (out / "splits_final.json").write_text(json.dumps(folds, indent=1))
    (out / "species_folds.json").write_text(json.dumps(fold_of, indent=1))
    print(f"cases ok: {len(cases)} from {len(sps)} species", flush=True)
