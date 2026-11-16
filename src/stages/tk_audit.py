"""Stage tk_audit: audit of the Toulkeridou et al. (2023) reference labels used to validate seg v3.

Why: for TK_063 the label in the deposited training.zip is rigidly offset from its image (~110 x 28 px on every slice);
shifted back it outlines the brain. This stage checks all 117 held-out TK cases in the same way.
Per case (fold model = the seg v3 fold whose validation set contains the case, exactly as nnU-Net validation: 2d,
mirroring, tile step 0.5, Gaussian):
1. full-volume prediction; Dice and relative volume error vs the stored label (must reproduce val_per_case.json);
2. rigid in-plane offset of the label: FFT cross-correlation of the z-projected masks -> (dy, dx); per-slice offsets
   (slices with both masks) -> median and IQR; Dice after the global shift;
3. z offset: best shift in [-zmax, zmax] after the in-plane shift.
Flag rule (fixed in the config before running): archive_offset if |shift| >= min_shift_px AND Dice gain >= min_gain AND
per-slice offset IQR <= max_iqr_px (a consistent rigid offset, not a shape disagreement).
Config: tk_raw_prefix, seg_v3 {fold: prefix}, min_shift_px, min_gain, max_iqr_px, zmax.
Outputs: audit.jsonl, preds/<case>.npz (packed prediction), summary.json.
"""
import json, pathlib, time
import numpy as np
import r2


def _shift2d(a, b):
    """(dy, dx) that maximises overlap of b rolled onto a (circular FFT cross-correlation)."""
    fa, fb = np.fft.fft2(a.astype(np.float64)), np.fft.fft2(b.astype(np.float64))
    cc = np.fft.ifft2(fa * np.conj(fb)).real
    dy, dx = np.unravel_index(int(cc.argmax()), cc.shape)
    H, W = a.shape
    return int(dy - H if dy > H // 2 else dy), int(dx - W if dx > W // 2 else dx)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 1.0


def main(cfg, out):
    import nibabel as nib, torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    c = r2.client()

    def fetch(key, dst):
        dst = pathlib.Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            c.download_file(r2.BUCKET, key, str(dst))
        return dst

    work = pathlib.Path("/root/tk_audit_cache"); M = work / "model"
    seg = {int(k): v for k, v in cfg["seg_v3"].items()}
    for fn in ("plans.json", "dataset.json"):
        fetch(seg[0] + fn, M / fn)
    (out / "preds").mkdir(exist_ok=True); rows = []
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device", dev, flush=True)
    raw = cfg["tk_raw_prefix"] + "nnUNet_raw/Dataset501_AntBrainTK/"
    zmax = int(cfg.get("zmax", 20))
    with open(out / "audit.jsonl", "w", encoding="utf-8") as fo:
        for f in sorted(seg):
            vpc = json.loads(fetch(seg[f] + "val_per_case.json", work / f"vpc_{f}.json").read_text())
            assert json.loads(fetch(seg[f] + "plans.json", work / f"plans_{f}.json").read_text()) == json.loads((M / "plans.json").read_text())
            fetch(seg[f] + "checkpoint_final.pth", M / f"fold_{f}" / "checkpoint_final.pth")
            p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True,
                                device=dev, verbose=False, allow_tqdm=False)
            p.initialize_from_trained_model_folder(str(M), use_folds=(f,), checkpoint_name="checkpoint_final.pth")
            for case, v in sorted(vpc.items()):
                if v.get("source") != "TK":
                    continue
                t0 = time.time()
                img = np.asarray(nib.load(str(fetch(raw + f"imagesTr/{case}_0000.nii.gz", work / "img" / f"{case}.nii.gz"))).dataobj).transpose(2, 1, 0).astype(np.float32)
                lab = np.asarray(nib.load(str(fetch(raw + f"labelsTr/{case}.nii.gz", work / "lab" / f"{case}.nii.gz"))).dataobj).transpose(2, 1, 0) > 0
                pr = p.predict_single_npy_array(img[None], {"spacing": [1.0, 1.0, 1.0]}, None, None, False) > 0
                np.savez_compressed(out / "preds" / f"{case}.npz", pred=np.packbits(pr), shape=np.array(pr.shape))
                d0 = dice(pr, lab)
                dy, dx = _shift2d(pr.sum(0), lab.sum(0))
                labs = np.roll(lab, (dy, dx), (1, 2)); d1 = dice(pr, labs)
                ks = [k for k in range(len(pr)) if pr[k].sum() > 50 and lab[k].sum() > 50]
                ks = ks[:: max(1, len(ks) // 40)]
                per = [_shift2d(pr[k], lab[k]) for k in ks]
                py = np.array([q[0] for q in per]) if per else np.array([0.0]); px = np.array([q[1] for q in per]) if per else np.array([0.0])
                iqr = float(max(np.subtract(*np.percentile(py, [75, 25])), np.subtract(*np.percentile(px, [75, 25]))))
                zbest, dz = 0, d1
                for z in range(-zmax, zmax + 1):
                    if z == 0:
                        continue
                    a = pr[max(0, z):len(pr) + min(0, z)]; b = labs[max(0, -z):len(labs) - max(0, z)]
                    dd = dice(a, b)
                    if dd > dz + 1e-4:
                        zbest, dz = z, dd
                shift = float(np.hypot(dy, dx)); gain = d1 - d0
                flag = shift >= cfg["min_shift_px"] and gain >= cfg["min_gain"] and iqr <= cfg["max_iqr_px"]
                rec = {"case": case, "fold": f, "species": v.get("species"), "dice_stored": v.get("dice"), "dice": round(d0, 4),
                       "rel_vol_err": round(float((pr.sum() - lab.sum()) / max(lab.sum(), 1)), 4),
                       "shift_dy": dy, "shift_dx": dx, "shift_px": round(shift, 1), "dice_after_shift": round(d1, 4),
                       "per_slice_shift_median": [float(np.median(py)), float(np.median(px))], "per_slice_shift_iqr": round(iqr, 1),
                       "z_shift": zbest, "dice_after_shift_z": round(dz, 4), "archive_offset_flag": bool(flag),
                       "label_frac": round(float(lab.mean()), 4), "pred_frac": round(float(pr.mean()), 4), "sec": round(time.time() - t0, 1)}
                rows.append(rec); fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec), flush=True)
    d = np.array([r["dice"] for r in rows]); fl = [r for r in rows if r["archive_offset_flag"]]
    keep = np.array([r["dice"] for r in rows if not r["archive_offset_flag"]])
    summ = {"n": len(rows), "reproduced_max_abs_diff": float(max(abs(r["dice"] - (r["dice_stored"] or 0)) for r in rows)),
            "median_dice": float(np.median(d)), "flagged": [r["case"] for r in fl],
            "median_dice_excluding_flagged": float(np.median(keep)) if len(keep) else None,
            "rule": {k: cfg[k] for k in ("min_shift_px", "min_gain", "max_iqr_px")}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
