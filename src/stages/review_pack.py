"""Stage review_pack: build the gate-1 human review package for AntScan brain drafts.

For every box of brain_boxes v2: draft A = nnInteractive v3 mask (upsampled to the box grid), draft B = scaled TK nnU-Net
mask post-processed to its largest 3D connected component (after a 1-voxel opening). Reports A-B Dice and auto-flags,
and writes a side-by-side JPEG per specimen (8 axial slices spanning the union; row A = nnInteractive in red,
row B = nnU-Net in cyan).
Config: boxes_run, nni_run, nnu_run, target_px, jpeg_width.
Outputs: review/<key>.jpg, masks_B/<key>.npy, review_index.json (list of per-specimen records), summary.json.
"""
import io, json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import r2


def largest(m):
    m = ndi.binary_opening(m > 0, iterations=1)
    lab, n = ndi.label(m)
    if n == 0: return m.astype(np.uint8)
    sizes = ndi.sum(m, lab, range(1, n + 1)); return (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def dice(a, b):
    a, b = a > 0, b > 0; return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def panel(box, A, B, path, width):
    u = (A > 0) | (B > 0); zz = np.nonzero(u.any((1, 2)))[0]
    ks = [int(v) for v in np.linspace(zz.min(), zz.max(), 10)[1:-1]] if len(zz) else [box.shape[0] // 2] * 8
    lo, hi = np.percentile(box, [1, 99.8])
    def tile(k, M, col):
        a = np.clip((box[k].astype(np.float32) - lo) / max(hi - lo, 1), 0, 1); rgb = np.stack([a] * 3, -1)
        mk = M[k] > 0; e = mk & ~ndi.binary_erosion(mk); rgb[e] = col
        return (rgb * 255).astype(np.uint8)
    rows = [np.concatenate([tile(k, A, [1, .15, .15]) for k in ks], 1), np.concatenate([tile(k, B, [.1, .9, 1]) for k in ks], 1)]
    im = Image.fromarray(np.concatenate(rows, 0)); im.thumbnail((width, width)); im.save(path, quality=82)


def main(cfg, out):
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    bp, ap, np_ = (f"ant-twin/runs/{cfg[k]}/outputs/" for k in ("boxes_run", "nni_run", "nnu_run"))
    man = [json.loads(l) for l in get(bp + "manifest.jsonl").decode().splitlines() if l.strip()]
    drafts = {json.loads(l)["key"]: json.loads(l) for l in get(ap + "drafts.jsonl").decode().splitlines() if l.strip()}
    (out / "review").mkdir(exist_ok=True); (out / "masks_B").mkdir(exist_ok=True)
    recs = []
    for m in [m for m in man if m.get("status") == "ok"]:
        key = m.get("key") or m["vial"]
        box = np.load(io.BytesIO(get(bp + f"boxes/{key}.npy")))
        B = largest(np.load(io.BytesIO(get(np_ + f"pred_{cfg['target_px']}/{key}.npy"))))
        d = drafts.get(key); A = np.zeros_like(B)
        if d:
            a = np.load(io.BytesIO(get(ap + f"masks/{key}.npy"))); ds = d["downsample"]
            a = np.repeat(np.repeat(np.repeat(a, ds, 0), ds, 1), ds, 2) if ds > 1 else a
            A[: min(a.shape[0], A.shape[0]), : min(a.shape[1], A.shape[1]), : min(a.shape[2], A.shape[2])] = a[: A.shape[0], : A.shape[1], : A.shape[2]]
        np.save(out / "masks_B" / f"{key}.npy", B)
        panel(box, A, B, out / "review" / f"{key}.jpg", cfg.get("jpeg_width", 1600))
        edgeB = bool(B[0].any() or B[-1].any() or B[:, 0].any() or B[:, -1].any() or B[:, :, 0].any() or B[:, :, -1].any())
        vox_um3 = float(m.get("voxel_um", 2.44)) ** 3
        r = {"key": key, "species": m["species"], "subfamily": m.get("subfamily"), "folder": m.get("folder"), "stained": m.get("stained"),
             "voxel_um": m.get("voxel_um"), "box_shape": list(box.shape), "dice_A_B": round(dice(A, B), 3),
             "vol_A_mm3": round(A.sum() * vox_um3 / 1e9, 5), "vol_B_mm3": round(B.sum() * vox_um3 / 1e9, 5),
             "frac_B": round(float(B.mean()), 4), "B_touches_edge": edgeB, "A_plausible": bool(d and any(x["plausible"] for x in d["candidates"]))}
        r["auto_flag"] = "agree" if r["dice_A_B"] >= 0.8 else ("B_only" if r["frac_B"] > 0.02 and not r["A_plausible"] else "check")
        recs.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    (out / "review_index.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    summ = {"n": len(recs), "agree_ge_0.8": sum(r["dice_A_B"] >= 0.8 for r in recs), "median_dice": float(np.median([r["dice_A_B"] for r in recs])),
            "flags": {f: sum(r["auto_flag"] == f for r in recs) for f in ("agree", "B_only", "check")}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
