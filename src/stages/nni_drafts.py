"""Stage nni_drafts: draft brain masks with nnInteractive (DKFZ; weights CC BY-NC-SA 4.0, used only for draft labels that a
human reviews) on full-resolution brain boxes produced by the brain_boxes stage (read from R2).
Prompt: one positive point at the box centre (the box was centred on the low-res brain proposal), optionally downsampled.
Config: source_run (R2 run id of brain_boxes), downsample, max_specimens.
Outputs: masks/<vial>.npy (uint8, same grid as the box after downsampling), qc/<vial>.png, drafts.jsonl.
"""
import os, io, json, time, pathlib
import numpy as np
from PIL import Image
import r2
from scipy import ndimage as ndi


def candidate_clicks(vol, k=5):
    """Centroids of smooth mid-grey components (brain neuropil is smooth soft tissue), nearest the box centre first."""
    v = ndi.gaussian_filter(vol.astype(np.float32), 1.0)
    mu = ndi.uniform_filter(v, 5); sd = np.sqrt(np.maximum(ndi.uniform_filter(v * v, 5) - mu * mu, 0))
    body = mu > np.percentile(mu, 35)
    lo, hi = np.percentile(mu[body], [20, 80])
    smooth = body & (sd < np.percentile(sd[body], 30)) & (mu > lo) & (mu < hi)
    smooth = ndi.binary_opening(smooth, iterations=2)
    lab, n = ndi.label(smooth)
    if n == 0:
        return [tuple(int(x // 2) for x in vol.shape)]
    sizes = ndi.sum(smooth, lab, range(1, n + 1)); ctr = np.array(vol.shape) / 2
    cands = []
    for i in np.argsort(sizes)[::-1][:20]:
        if sizes[i] < 0.002 * vol.size: break
        com = np.array(ndi.center_of_mass(smooth, lab, i + 1))
        pts = np.argwhere(lab == i + 1); p = pts[np.argmin(((pts - com) ** 2).sum(1))]  # a voxel inside the component
        cands.append((float(np.linalg.norm((p - ctr) / ctr)), tuple(int(x) for x in p)))
    cands.sort()
    out = [tuple(int(x // 2) for x in vol.shape)] + [c for _, c in cands]
    return out[:k]


def overlay(vol, m, path):
    zz = np.nonzero(m.any((1, 2)))[0]
    ks = [int(v) for v in np.linspace(zz.min(), zz.max(), 8)] if len(zz) else [vol.shape[0] // 2] * 8
    lo, hi = np.percentile(vol, [1, 99.8])
    def ov(k):
        a = np.clip((vol[k].astype(np.float32) - lo) / max(hi - lo, 1), 0, 1); rgb = np.stack([a, a, a], -1)
        mk = m[k] > 0; edge = mk & ~(np.roll(mk, 1, 0) & np.roll(mk, -1, 0) & np.roll(mk, 1, 1) & np.roll(mk, -1, 1))
        rgb[mk] = rgb[mk] * 0.75 + np.array([0.25, 0, 0]); rgb[edge] = [1, 0.1, 0.1]
        return (rgb * 255).astype(np.uint8)
    t = [ov(k) for k in ks]
    M = np.concatenate([np.concatenate(t[:4], 1), np.concatenate(t[4:], 1)], 0)
    im = Image.fromarray(M); im.thumbnail((2000, 2000)); im.save(path)


def main(cfg, out):
    import torch
    from nnInteractive.model_management import ensure_model_available, get_default_model_id
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    mid = get_default_model_id(); mp = ensure_model_available(mid); print("model", mid)
    s = nnInteractiveInferenceSession(device=dev, use_torch_compile=False, verbose=False, torch_n_threads=os.cpu_count(), do_autozoom=True)
    s.initialize_from_trained_model_folder(str(mp))
    c = r2.client(); pre = f"ant-twin/runs/{cfg['source_run']}/outputs/"
    man = [json.loads(l) for l in c.get_object(Bucket=r2.BUCKET, Key=pre + "manifest.jsonl")["Body"].read().decode().splitlines() if l.strip()]
    ok = [m for m in man if m.get("status") == "ok"][: cfg.get("max_specimens", 999)]
    (out / "masks").mkdir(exist_ok=True); (out / "qc").mkdir(exist_ok=True)
    ds = int(cfg.get("downsample", 1))
    with open(out / "drafts.jsonl", "w", encoding="utf-8") as f:
        for m in ok:
            t0 = time.time(); v = m["vial"]
            box = np.load(io.BytesIO(c.get_object(Bucket=r2.BUCKET, Key=pre + f"boxes/{v}.npy")["Body"].read()))
            vol = box[::ds, ::ds, ::ds] if ds > 1 else box
            s.set_image(vol[None].astype(np.uint8))
            best = None; tried = []
            for ctr in candidate_clicks(vol, cfg.get("n_clicks", 5)):
                s.reset_interactions(); s.set_image(vol[None].astype(np.uint8))
                tgt = torch.zeros(vol.shape, dtype=torch.uint8); s.set_target_buffer(tgt)
                s.add_point_interaction(ctr, include_interaction=True)
                cm = tgt.numpy().astype(np.uint8).copy(); frac = cm.sum() / cm.size
                edge = int(cm[0].sum() + cm[-1].sum() + cm[:, 0].sum() + cm[:, -1].sum() + cm[:, :, 0].sum() + cm[:, :, -1].sum())
                ok_ = bool(0.01 <= frac <= 0.35)
                score = (ok_, -edge / max(cm.sum(), 1), cm.sum())
                tried.append({"click": ctr, "vox": int(cm.sum()), "frac": round(float(frac), 4), "edge_vox": edge, "plausible": ok_})
                if best is None or score > best[0]: best = (score, ctr, cm)
            _, ctr, mk = best
            np.save(out / "masks" / f"{v}.npy", mk); overlay(vol, mk, out / "qc" / f"{v}.png")
            touches = bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any() or mk[:, :, 0].any() or mk[:, :, -1].any())
            rec = {"vial": v, "species": m["species"], "box_shape": list(box.shape), "downsample": ds, "click": ctr,
                   "mask_vox": int(mk.sum()), "candidates": tried, "mask_vox_fullres_est": int(mk.sum()) * ds ** 3, "touches_box_edge": touches, "sec": round(time.time() - t0, 1)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); print(json.dumps(rec, ensure_ascii=False))
            s.reset_interactions()
