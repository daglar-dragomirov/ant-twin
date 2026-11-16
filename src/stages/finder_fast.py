"""Stage finder_fast: brain localisation on whole-specimen low-res stacks with the trained finder, run in-process.

Why: nnUNetv2_predict spends ~90 s per specimen resampling huge inputs/outputs with CPU splines. Here the model is
loaded once; each cached low-res stack is cropped to the specimen, normalised as nnU-Net CTNormalization, resampled
to the plan spacing with trilinear interpolation on the GPU, predicted with the sliding window (no mirroring); the
largest predicted component's bbox is converted to low-res and full-res coordinates arithmetically.
Config: specimens_run, model_run, dataset_id, dataset_name, configuration, fold, trainer, margin_frac, crop_margin,
        cache_prefix, skip_runs, shard, workdir, qc_every, castes (default ["worker"]), require_colony (default true).
Outputs: finder.jsonl (same fields as finder_predict: key, status, box_zyx, lowres_bbox, candidates, ...), qc/, summary.json.
"""
import io, os, json, time, pathlib
import numpy as np
from scipy import ndimage as ndi
import r2
from stages.tk_volumes import qc


def vial_code(v):
    p = str(v).split("-")
    return "%02d-%02d" % (int(p[0]), int(p[1])) if len(p) == 2 and all(x.isdigit() for x in p) else str(v)


def crop_to_specimen(v, margin):
    from skimage.filters import threshold_multiotsu
    t_hi = float(threshold_multiotsu(v[::4, ::4, ::4].astype(np.float32), classes=3)[-1])
    b = ndi.binary_opening(v[::2, ::2, ::2] > t_hi, iterations=1); lab, n = ndi.label(b)
    if n == 0: return v, [0, 0, 0]
    sz = ndi.sum(b, lab, range(1, n + 1)); keep = np.isin(lab, [i + 1 for i in range(n) if sz[i] >= 0.01 * sz.max()])
    border = set(np.unique(np.concatenate([lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))) - {0}
    for i, sl in enumerate(ndi.find_objects(lab)):
        if sl is None: continue
        if ((sl[1].stop - sl[1].start) > 0.8 * b.shape[1] and (sl[2].stop - sl[2].start) > 0.8 * b.shape[2]) or (i + 1) in border: keep[lab == i + 1] = False
    if not keep.any(): return v, [0, 0, 0]
    zz, yy, xx = np.nonzero(keep)
    z0, y0, x0 = max(0, zz.min() * 2 - margin), max(0, yy.min() * 2 - margin), max(0, xx.min() * 2 - margin)
    z1, y1, x1 = min(v.shape[0], zz.max() * 2 + 2 + margin), min(v.shape[1], yy.max() * 2 + 2 + margin), min(v.shape[2], xx.max() * 2 + 2 + margin)
    return v[z0:z1, y0:y1, x0:x1], [int(z0), int(y0), int(x0)]


def main(cfg, out):
    import torch, pandas as pd
    import torch.nn.functional as F
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    W = pathlib.Path(cfg.get("workdir", "/root/nnunet")); c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    name = f"Dataset{cfg['dataset_id']}_{cfg['dataset_name']}"; tr, conf, fold = cfg["trainer"], cfg["configuration"], cfg["fold"]
    R = W / "res_fast" / name / f"{tr}__nnUNetPlans__{conf}"; (R / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
    mp = f"ant-twin/runs/{cfg['model_run']}/outputs/"
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"): c.download_file(r2.BUCKET, mp + f, str(R / f))
    c.download_file(r2.BUCKET, mp + "checkpoint_final.pth", str(R / f"fold_{fold}" / "checkpoint_final.pth"))
    dev = torch.device("cuda")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True, device=dev, verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(R), use_folds=(fold,), checkpoint_name="checkpoint_final.pth")
    tf = list(pred.plans_manager.transpose_forward); tb = list(pred.plans_manager.transpose_backward)
    tsp_t = np.array(pred.configuration_manager.spacing, dtype=float)  # plan spacing, in transposed axis order
    tsp = tsp_t[tb]                                                      # back to our (z, y, x) order
    ip = json.loads((R / "plans.json").read_text())["foreground_intensity_properties_per_channel"]["0"]
    lo_i, hi_i, mu_i, sd_i = ip["percentile_00_5"], ip["percentile_99_5"], ip["mean"], max(ip["std"], 1e-8)
    print("transpose_forward", tf, "target spacing (zyx)", tsp.tolist(), "intensity", lo_i, hi_i, mu_i, sd_i, flush=True)
    d = pd.read_csv(io.BytesIO(get(f"ant-twin/runs/{cfg['specimens_run']}/outputs/specimens.csv")))
    d = d[d.caste.isin(cfg.get("castes", ["worker"])) & d.url_full.notna()]
    if cfg.get("require_colony", True): d = d[d.colony_any.notna()]
    recs = d.to_dict("records"); si, sn = cfg.get("shard", [0, 1]); recs = recs[si::sn]
    vf = cfg.get("voxel_filter")  # {"mode": "only"|"exclude", "voxels": [6.11]}: split slow (upsampled) scans to another pod
    if vf:
        isv = lambda r: any(abs(float(r["voxel_um"]) - v) < 0.01 for v in vf["voxels"])
        recs = [r for r in recs if (isv(r) if vf["mode"] == "only" else not isv(r))]
    def done_elsewhere():
        ks = set()
        for run in cfg.get("skip_runs", []):
            try:
                for l in get(f"ant-twin/runs/{run}/outputs/finder.jsonl").decode().splitlines():
                    if l.strip(): ks.add(json.loads(l)["key"])
            except Exception: pass
        return ks
    keyof = lambda r: f"{r['folder']}__{vial_code(r['vial'])}"
    TP = json.loads(get(cfg["true_pages_key"])) if cfg.get("true_pages_key") else {}
    if cfg.get("only_keys_key"):  # v8: restrict to a key list (specimens still without an approved brain)
        only = set(json.loads(get(cfg["only_keys_key"]))); recs = [r for r in recs if keyof(r) in only]
    skip = done_elsewhere(); recs = [r for r in recs if keyof(r) not in skip]; print(f"to do: {len(recs)}", flush=True)
    (out / "qc").mkdir(exist_ok=True); n_ok = 0; t_all = time.time()
    with open(out / "finder.jsonl", "w", encoding="utf-8") as fo:
        for i, r in enumerate(recs):
            key = keyof(r); t0 = time.time()
            base = {"key": key, "species": r["name"], "species_key": r["species_key"], "subfamily": r["subfamily"], "stained": r["stained"],
                    "voxel_um": r["voxel_um"], "folder": r["folder"], "vial": r["vial"], "url_full": r["url_full"]}
            if i % 20 == 0 and key in done_elsewhere(): continue
            try:
                z = np.load(io.BytesIO(get(cfg.get("cache_prefix", "ant-twin/cache/lowres/") + key + ".npz")))
                v, nlow, P, H, Wd = z["img"], int(z["n_lowres"]), int(z["pages"]), int(z["H"]), int(z["W"])
                P_est = P; P = int(TP[key]["pages"]) if key in TP else P  # v8: exact page count (size-based estimate wrong for compressed stacks)
            except Exception as ex:
                fo.write(json.dumps({**base, "status": "not_cached"}) + "\n"); continue
            vox = float(r["voxel_um"]); f = np.array([P / nlow, H / v.shape[1], Wd / v.shape[2]]); sp1 = vox * f  # um per low-res voxel
            vc, off = crop_to_specimen(v, cfg.get("crop_margin", 40))
            vt, sp1t = np.array(vc.shape)[tf], sp1[tf]
            new_shape_t = [max(8, int(round(s_ * a / b_))) for s_, a, b_ in zip(vt, sp1t, tsp_t)]
            nvox = int(np.prod(new_shape_t))
            huge = nvox > cfg.get("huge_voxels", 2e8)              # accumulate logits on CPU
            cpu_interp = huge or nvox > cfg.get("cpu_interp_voxels", 1.5e9)  # CUDA trilinear fails beyond ~2^31 elements
            x = torch.from_numpy(np.ascontiguousarray(vc.transpose(tf)).astype(np.float32))
            if not cpu_interp: x = x.to(dev)
            x = ((x.clamp(lo_i, hi_i) - mu_i) / sd_i)[None, None]
            with torch.no_grad():
                x = F.interpolate(x, size=new_shape_t, mode="trilinear", align_corners=False)[0]
                if cpu_interp and not huge: x = x.to(dev); base["cpu_interp"] = True
                if huge: pred.perform_everything_on_device = False; base["cpu_path"] = True
                try:
                    logits = pred.predict_sliding_window_return_logits(x)
                except torch.OutOfMemoryError:  # very large (upsampled) volume: accumulate logits on the CPU instead
                    torch.cuda.empty_cache(); pred.perform_everything_on_device = False
                    x = x.to("cpu"); logits = pred.predict_sliding_window_return_logits(x)
                    pred.perform_everything_on_device = True; base["oom_fallback"] = True
                pred.perform_everything_on_device = True
                seg = (logits[1] > logits[0]).to("cpu").numpy().astype(np.uint8).transpose(tb) if logits.shape[0] == 2 else logits.argmax(0).to("cpu").numpy().astype(np.uint8).transpose(tb)  # v8b: no int64 argmax tensor
            new_shape = list(seg.shape)
            del x, logits; torch.cuda.empty_cache()
            lab, n = ndi.label(seg)
            rec = {**base, "P": P, "P_est": P_est, "H": H, "W": Wd, "nlow": nlow, "lowres_shape": list(v.shape), "crop_offset": off, "plan_shape": new_shape, "n_components": int(n)}
            if n == 0:
                rec["status"] = "no_brain"; fo.write(json.dumps(rec) + "\n"); fo.flush(); continue
            sizes = ndi.sum(seg, lab, range(1, n + 1)); order = np.argsort(sizes)[::-1]
            scale = tsp / sp1  # low-res voxels per plan voxel
            def bbox_lowres(mask):
                zz, yy, xx = np.nonzero(mask)
                lo = np.array([zz.min(), yy.min(), xx.min()]) * scale + off; hi = (np.array([zz.max(), yy.max(), xx.max()]) + 1) * scale + off
                return lo, hi
            cand = []
            for ci in order[:3]:
                l_, h_ = bbox_lowres(lab == ci + 1); cand.append({"vox": int(sizes[ci]), "lowres_bbox": [int(a) for a in (*l_, *h_)]})
            lo, hi = bbox_lowres(lab == order[0] + 1)
            c_ = (lo + hi) / 2 * f; half = (hi - lo) / 2 * f * (1 + cfg.get("margin_frac", 0.6)); lim = np.array([P, H, Wd])
            z0, y0, x0 = np.maximum(np.floor(c_ - half), 0).astype(int); z1, y1, x1 = np.minimum(np.ceil(c_ + half), lim).astype(int)
            rec.update(status="ok", lowres_bbox=[int(a) for a in (*lo, *hi)], box_zyx=[int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                       brain_vol_lowres_um3=float(sizes[order[0]] * np.prod(tsp)), largest_comp_frac=round(float(sizes.max() / sizes.sum()), 4),
                       candidates=cand, sec=round(time.time() - t0, 1))
            fo.write(json.dumps(rec) + "\n"); fo.flush(); n_ok += 1
            if n_ok % cfg.get("qc_every", 20) == 0:
                big = (lab == order[0] + 1).astype(np.uint8); zz = np.nonzero(big.any((1, 2)))[0]; k = int(np.median(zz))
                vv = ndi.zoom(vc, [a / b for a, b in zip(seg.shape, vc.shape)], order=1)
                sl = slice(max(0, k - 30), min(seg.shape[0], k + 30)); qc(vv[sl], big[sl], out / "qc" / f"{key}.png")
            if i % 10 == 0: print(f"{i}/{len(recs)} ok={n_ok} last {rec['sec']}s avg {(time.time() - t_all) / (i + 1):.1f}s", flush=True)
    rows = [json.loads(l) for l in open(out / "finder.jsonl", encoding="utf-8")]
    from collections import Counter
    summ = {"n": len(rows), "status": dict(Counter(x.get("status") for x in rows)), "sec_total": round(time.time() - t_all)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
