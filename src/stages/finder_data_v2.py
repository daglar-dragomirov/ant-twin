"""Stage finder_data_v2 (v3 node): training set for the final low-res brain finder.

v3: z factor and spacing use the exact page counts from meta/true_pages.json; gate-1 cases are kept only where the
boxes-v2 manifest page count equals the true count.

Cases = (a) all cases of an earlier finder dataset run (gate-1, 48), copied as is; (b) every specimen whose full-res brain
mask (brain_masks_v1/v2 caches) was rated good by the vision agents with the box on the head and the brain fully inside
the box; v2 preferred over v1. Each mask is mapped onto the cached low-res stack with the scan geometry, then image and
label are downsampled by ds; spacing is physical. Species-level folds.
Config: old_dataset_run, old_dataset_id, old_dataset_name, reviews [{files:[...], mask_prefix}], specimens_run,
        cache_prefix, ds, dataset_id, dataset_name, n_folds, seed.
Outputs: nnUNet_raw/Dataset<id>_<name>/{imagesTr,labelsTr,dataset.json}, splits_final.json, cases.jsonl, qc/.
"""
import io, json, random
import numpy as np
from scipy import ndimage as ndi
import r2
from stages.tk_volumes import qc


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)



def to_iso(img, lab, sp, iso):
    """Resample image (linear) and label (linear > 0.5) from spacing sp (per axis, um) to isotropic iso um."""
    f = [s_ / iso for s_ in sp]
    return ndi.zoom(img, f, order=1), (ndi.zoom(lab.astype(np.float32), f, order=1) > 0.5).astype(np.uint8)


def main(cfg, out):
    import nibabel as nib, pandas as pd
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    D = out / "nnUNet_raw" / f"Dataset{cfg['dataset_id']}_{cfg['dataset_name']}"; (D / "imagesTr").mkdir(parents=True, exist_ok=True); (D / "labelsTr").mkdir(exist_ok=True)
    (out / "qc").mkdir(exist_ok=True)
    sp_tab = pd.read_csv(io.BytesIO(get(f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")))
    species_of = {f"{r.folder}__{vial_code(r.vial)}": r.name for r in sp_tab.itertuples()}
    cases = {}  # case -> species
    tp = json.loads(get(cfg["true_pages_key"]))  # exact full-res page counts (size-based estimate is wrong for compressed stacks)
    man = {}
    for l in get(f"ant-twin/runs/{cfg['old_boxes_run']}/outputs/manifest.jsonl").decode().splitlines():
        if l.strip(): m = json.loads(l); man[m.get("key") or m["vial"]] = m
    dropped_old = []
    # (a) copy the old dataset
    op = f"ant-twin/runs/{cfg['old_dataset_run']}/outputs/"
    old_keys = set()
    for l in get(op + "cases.jsonl").decode().splitlines():
        r = json.loads(l)
        if r.get("status") != "ok" or r.get("label_vox", 0) == 0: continue
        mp = man.get(r["key"], {}).get("n_pages"); tpk = tp.get(r["key"], {}).get("pages")
        if mp is None or tpk is None or int(mp) != int(tpk): dropped_old.append({"key": r["key"], "manifest_pages": mp, "true_pages": tpk}); continue
        for sub, suf in (("imagesTr", "_0000.nii.gz"), ("labelsTr", ".nii.gz")):
            src = op + f"nnUNet_raw/Dataset{cfg['old_dataset_id']}_{cfg['old_dataset_name']}/{sub}/{r['case']}{suf}"
            (D / sub / f"{r['case']}{suf}").write_bytes(get(src))
        if cfg.get("iso_um"):
            fi, fl = D / "imagesTr" / f"{r['case']}_0000.nii.gz", D / "labelsTr" / f"{r['case']}.nii.gz"
            ni, nl = nib.load(str(fi)), nib.load(str(fl)); spx = [abs(float(a)) for a in np.diag(ni.affine)[:3]]
            a, b = to_iso(np.asarray(ni.dataobj), np.asarray(nl.dataobj), spx, cfg["iso_um"]); af = np.diag([cfg["iso_um"]] * 3 + [1.0])
            nib.save(nib.Nifti1Image(np.ascontiguousarray(a), af), fi); nib.save(nib.Nifti1Image(np.ascontiguousarray(b), af), fl)
        cases[r["case"]] = r["species"]; old_keys.add(r["key"])
    print(f"copied {len(cases)} old cases; dropped {len(dropped_old)} with a wrong page count", flush=True)
    (out / "dropped_old.json").write_text(json.dumps(dropped_old, indent=1))
    # (b) agent-approved masks; later review sets override earlier ones for the same key
    chosen = {}
    unchanged = lambda k: k in tp and tp[k]["pages"] == tp[k]["pages_est"]
    for rv in cfg["reviews"]:
        for f in rv["files"]:
            for x in json.loads(get(f)):
                base = x["key"].split("@")[0]
                if rv.get("unchanged_only") and not unchanged(base): continue  # v4: boxes_v3 reviews hold only where P was right
                ok = x.get("mask") == "good" and x.get("box_on_head") is True and x.get("brain_fully_in_box") is not False
                if ok: chosen[x["key"]] = rv["mask_prefix"]
                elif x["key"] in chosen and rv.get("override_bad", True): chosen.pop(x["key"])  # a later review of the same box wins
    # one case per specimen: tiny-ant end keys (key@A/@B) map onto the base key's low-res stack
    by_base = {}
    for k in sorted(chosen):
        by_base.setdefault(k.split("@")[0], k)
    chosen = {k: chosen[k] for b, k in by_base.items() if b not in old_keys}
    print(f"new approved specimens: {len(chosen)}", flush=True)
    ds = int(cfg.get("ds", 2)); recs = []
    for i, (key, mpre) in enumerate(sorted(chosen.items())):
        try:
            lk = key.split("@")[0]
            z = np.load(io.BytesIO(get(cfg["cache_prefix"] + lk + ".npz"))); mz = np.load(io.BytesIO(get(mpre + key + ".npz")))
        except Exception as ex:
            recs.append({"key": key, "status": "missing", "error": repr(ex)[:120]}); continue
        v, nlow, H, W = z["img"], int(z["n_lowres"]), int(z["H"]), int(z["W"]); vox = float(z["voxel_um"])
        if lk not in tp: recs.append({"key": key, "status": "no_true_pages"}); continue
        P = int(tp[lk]["pages"])
        mk, bz = mz["mask"], [int(a) for a in mz["box_zyx"]]
        f = np.array([P / nlow, H / v.shape[1], W / v.shape[2]])
        z0, z1, y0, y1, x0, x1 = bz
        tz0, ty0, tx0 = int(round(z0 / f[0])), int(round(y0 / f[1])), int(round(x0 / f[2]))
        tgt = [max(1, int(round(mk.shape[k] / f[k]))) for k in range(3)]
        small = ndi.zoom(mk.astype(np.float32), [t / s for t, s in zip(tgt, mk.shape)], order=1) > 0.5
        lab = np.zeros(v.shape, np.uint8)
        sz = [min(small.shape[0], v.shape[0] - tz0), min(small.shape[1], v.shape[1] - ty0), min(small.shape[2], v.shape[2] - tx0)]
        if min(sz) <= 0: recs.append({"key": key, "status": "out_of_stack"}); continue
        lab[tz0: tz0 + sz[0], ty0: ty0 + sz[1], tx0: tx0 + sz[2]] = small[: sz[0], : sz[1], : sz[2]]
        if cfg.get("iso_um"): img, lab = to_iso(v, lab, list(vox * f), cfg["iso_um"])  # v5: isotropic, no nnU-Net resampling
        else: img, lab = v[::ds, ::ds, ::ds], lab[::ds, ::ds, ::ds]
        if lab.sum() == 0: recs.append({"key": key, "status": "empty_label"}); continue
        sp = (cfg["iso_um"],) * 3 if cfg.get("iso_um") else (vox * f[0] * ds, vox * f[1] * ds, vox * f[2] * ds); case = f"LR2_{i:04d}"; aff = np.diag([sp[2], sp[1], sp[0], 1.0])
        nib.save(nib.Nifti1Image(np.ascontiguousarray(img.transpose(2, 1, 0)), aff), D / "imagesTr" / f"{case}_0000.nii.gz")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(lab.transpose(2, 1, 0)), aff), D / "labelsTr" / f"{case}.nii.gz")
        cases[case] = species_of.get(lk, lk)
        recs.append({"key": key, "case": case, "species": cases[case], "status": "ok", "label_vox": int(lab.sum()), "spacing_um": [round(s, 3) for s in sp], "mask_source": mpre, "pages_true": P, "pages_est": int(z["pages"])})
        if i % 15 == 0:
            zz = np.nonzero(lab.any((1, 2)))[0]; k = int(zz[len(zz) // 2]); sl = slice(max(0, k - 40), min(img.shape[0], k + 40))
            qc(img[sl], lab[sl], out / "qc" / f"{case}_{key}.png")
        print(json.dumps(recs[-1]), flush=True)
    with open(out / "cases.jsonl", "w", encoding="utf-8") as fo:
        for r in recs: fo.write(json.dumps(r) + "\n")
    (D / "dataset.json").write_text(json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "brain": 1}, "numTraining": len(cases),
                                                "file_ending": ".nii.gz", "description": "AntScan low-res stacks; brains from gate-1 + agent-approved seg masks"}, indent=1))
    sps = sorted(set(cases.values())); random.Random(cfg["seed"]).shuffle(sps); k = cfg["n_folds"]; fo_ = {s: j % k for j, s in enumerate(sps)}
    folds = [{"train": sorted(cs for cs, s in cases.items() if fo_[s] != fi), "val": sorted(cs for cs, s in cases.items() if fo_[s] == fi)} for fi in range(k)]
    (out / "splits_final.json").write_text(json.dumps(folds, indent=1))
    (out / "summary.json").write_text(json.dumps({"cases": len(cases), "new_ok": sum(r["status"] == "ok" for r in recs), "old_dropped": len(dropped_old), "species": len(sps)}, indent=1))
    print("cases", len(cases), "species", len(sps), flush=True)
