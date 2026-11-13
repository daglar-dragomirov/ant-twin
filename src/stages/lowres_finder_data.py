"""Stage lowres_finder_data: nnU-Net dataset for finding the brain on whole-specimen AntScan low-res stacks.

Labels come from the automated gate-1 review (3 vision agents): drafts rated good, or "fix" (mostly brain), on boxes
of brain_boxes v2. Each chosen full-res box mask is mapped into the low-res stack grid using the scan geometry
(full-res pages/H/W vs low-res slices/h/w and the box offset), then the stack and label are downsampled by `ds`.
Spacing is physical (um) per axis, so 1.22/2.44/6.11 um scans mix correctly.
Config: boxes_run, nni_run, review_run, review_key (R2 key of auto_review_summary.json), review_files (list of R2 keys
        of auto_review_N.json), ds, dataset_id, n_folds, seed, workers.
Outputs: nnUNet_raw/Dataset<id>_AntBrainLowres/{imagesTr,labelsTr,dataset.json}, splits_final.json, cases.jsonl, qc/.
"""
import io, json, random, collections, concurrent.futures as cf
import numpy as np
from scipy import ndimage as ndi
import r2
import bigtiff_range as BT
from stages.brain_boxes import lowres_stack
from stages.tk_volumes import qc


def main(cfg, out):
    import nibabel as nib
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    bp, ap, rp = (f"ant-twin/runs/{cfg[k]}/outputs/" for k in ("boxes_run", "nni_run", "review_run"))
    man = {(m.get("key") or m["vial"]): m for m in (json.loads(l) for l in get(bp + "manifest.jsonl").decode().splitlines() if l.strip())}
    sel = {e["key"]: e for e in json.loads(get(bp + "selection.json"))}
    drafts = {json.loads(l)["key"]: json.loads(l) for l in get(ap + "drafts.jsonl").decode().splitlines() if l.strip()}
    rev = {}
    for k in cfg["review_files"]:
        for x in json.loads(get(k)): rev[x["key"]] = x
    chosen = {}
    for k, x in rev.items():
        if x["verdict"] in ("A", "both"): chosen[k] = "A"
        elif x["verdict"] == "B": chosen[k] = "B"
        elif x["verdict"] == "fix": chosen[k] = "A" if x["A"] in ("good", "fix") else ("B" if x["B"] in ("good", "fix") else None)
    chosen = {k: v for k, v in chosen.items() if v and rev[k].get("box_on_head") is not False}
    print(f"labelled specimens: {len(chosen)} ({collections.Counter(chosen.values())})", flush=True)
    D = out / "nnUNet_raw" / f"Dataset{cfg['dataset_id']}_AntBrainLowres"; (D / "imagesTr").mkdir(parents=True, exist_ok=True); (D / "labelsTr").mkdir(exist_ok=True)
    (out / "qc").mkdir(exist_ok=True); ds = int(cfg.get("ds", 2))

    def one(i_key):
        i, key = i_key; m, e = man[key], sel[key]
        box = np.load(io.BytesIO(get(bp + f"boxes/{key}.npy"))).shape
        if chosen[key] == "A":
            a = np.load(io.BytesIO(get(ap + f"masks/{key}.npy"))); d = drafts[key]["downsample"]
            mk = np.repeat(np.repeat(np.repeat(a, d, 0), d, 1), d, 2)[: box[0], : box[1], : box[2]] if d > 1 else a
        else:
            mk = np.load(io.BytesIO(get(rp + f"masks_B/{key}.npy")))
        info = BT.layout(m["url"]); H, W, P = info["H"], info["W"], m["n_pages"]
        v, nlow = lowres_stack(m["vial"], 1, e["lowres_base"])
        if v is None: return {"key": key, "status": "no_lowres"}
        fz, fy, fx = P / nlow, H / v.shape[1], W / v.shape[2]
        z0, z1, y0, y1, x0, x1 = m["box_zyx"]
        lab = np.zeros(v.shape, np.uint8)
        tz0, ty0, tx0 = int(round(z0 / fz)), int(round(y0 / fy)), int(round(x0 / fx))
        tgt = (max(1, int(round(mk.shape[0] / fz))), max(1, int(round(mk.shape[1] / fy))), max(1, int(round(mk.shape[2] / fx))))
        small = ndi.zoom(mk.astype(np.float32), [t / s for t, s in zip(tgt, mk.shape)], order=1) > 0.5
        sz = [min(small.shape[0], v.shape[0] - tz0), min(small.shape[1], v.shape[1] - ty0), min(small.shape[2], v.shape[2] - tx0)]
        lab[tz0: tz0 + sz[0], ty0: ty0 + sz[1], tx0: tx0 + sz[2]] = small[: sz[0], : sz[1], : sz[2]]
        img = v[::ds, ::ds, ::ds]; lab = lab[::ds, ::ds, ::ds]
        vox = float(m.get("voxel_um", 2.44)); sp = (vox * fz * ds, vox * fy * ds, vox * fx * ds)
        case = f"LR_{i:03d}"
        aff = np.diag([sp[2], sp[1], sp[0], 1.0])
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), aff), D / "imagesTr" / f"{case}_0000.nii.gz")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(lab.transpose(2, 1, 0)), aff), D / "labelsTr" / f"{case}.nii.gz")
        zz = np.nonzero(lab.any((1, 2)))[0]
        if len(zz):
            k = int(zz[len(zz) // 2]); sl = slice(max(0, k - 40), min(img.shape[0], k + 40))
            qc(img[sl], lab[sl], out / "qc" / f"{case}_{key}.png")
        return {"key": key, "case": case, "species": m["species"], "draft": chosen[key], "verdict": rev[key]["verdict"], "status": "ok",
                "lowres_shape": list(v.shape), "shape": list(img.shape), "spacing_um": [round(s, 3) for s in sp], "label_vox": int(lab.sum())}

    recs = []
    with cf.ThreadPoolExecutor(cfg.get("workers", 6)) as ex, open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        for r in ex.map(one, enumerate(sorted(chosen))):
            recs.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); print(json.dumps(r, ensure_ascii=False), flush=True)
    ok = [r for r in recs if r["status"] == "ok" and r["label_vox"] > 0]
    (D / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "brain": 1}, "numTraining": len(ok),
                                                "file_ending": ".nii.gz", "description": "AntScan low-res stacks with auto-reviewed brain labels"}, indent=1))
    for r in recs:
        if r["status"] == "ok" and r["label_vox"] == 0:
            for sub in ("imagesTr", "labelsTr"):
                for p in (D / sub).glob(r["case"] + "*"): p.unlink()
    sps = sorted({r["species"] for r in ok}); random.Random(cfg["seed"]).shuffle(sps); k = cfg["n_folds"]
    fo = {s: j % k for j, s in enumerate(sps)}
    folds = [{"train": sorted(r["case"] for r in ok if fo[r["species"]] != fi), "val": sorted(r["case"] for r in ok if fo[r["species"]] == fi)} for fi in range(k)]
    (out / "splits_final.json").write_text(json.dumps(folds, indent=1)); print(f"cases ok: {len(ok)}", flush=True)
