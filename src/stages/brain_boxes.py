"""Stage brain_boxes: pick a stratified training set, propose a brain box on the low-res stack, fetch that box from the
full-resolution BigTIFF via HTTP Range, and write QC images. Config keys (configs/experiment.json):
  n_specimens, seed, lowres_zstep, pad_frac, max_box_um, workers
Outputs: boxes/<vial>.npy (uint8 full-res box), qc/<vial>.png, manifest.jsonl (one line per specimen).
"""
import json, random, hashlib, time, io, urllib.request, concurrent.futures as cf, pathlib
import numpy as np
from PIL import Image
import bigtiff_range as BT
import lowres_seg as LS

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}
ROOT = "https://biomedisa.info/media/antscan/processed/5x"


def vial_code(v):
    a, b = v.split("-"); return "%02d-%02d" % (int(a), int(b))


def http_ok(url):
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=40) as r: return r.status in (200, 206)
    except Exception:
        return False


def lowres_stack(vial, zstep):
    base = f"{ROOT}/{vial}_lowres/slice_%04d.png"
    if not http_ok(base % 0): return None, 0
    lo, hi = 0, 256
    while hi < 20000 and http_ok(base % hi): lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if http_ok(base % mid) else (lo, mid)
    n = lo + 1
    def get(i):
        for k in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(base % i, headers=UA), timeout=60) as r:
                    return i, np.asarray(Image.open(io.BytesIO(r.read())).convert("L"))
            except Exception: time.sleep(1 + k)
        return i, None
    with cf.ThreadPoolExecutor(8) as ex:
        sl = dict(ex.map(get, range(0, n, zstep)))
    if any(a is None for a in sl.values()): return None, n
    return np.stack([sl[i] for i in sorted(sl)]), n


def select(eligible, n, seed):
    rnd = random.Random(seed)
    by_sp = {}
    for e in eligible:
        if abs(e.get("voxel", 0) - 2.44) < 0.01: by_sp.setdefault(e["species"], e)
    sp = list(by_sp.values()); rnd.shuffle(sp)
    by_sf = {}
    for e in sp: by_sf.setdefault(e["subfamily"], []).append(e)
    for v in by_sf.values(): v.sort(key=lambda e: e["body_um3"])
    out = []
    while len(out) < n and any(by_sf.values()):  # round-robin over subfamilies, alternating small/large bodies
        for k in sorted(by_sf):
            if by_sf[k] and len(out) < n:
                out.append(by_sf[k].pop(0 if len(out) % 2 else -1))
    return out


def qc_png(box, path):
    z, y, x = [s // 2 for s in box.shape]
    tiles = [box[z], box[:, y, :], box[:, :, x]]
    h = max(t.shape[0] for t in tiles)
    row = np.concatenate([np.pad(t, ((0, h - t.shape[0]), (0, 4))) for t in tiles], 1)
    img = Image.fromarray(row); img.thumbnail((1400, 600)); img.save(path)


def one(e, cfg, out):
    vial = vial_code(e["vial"]); rec = {"vial": vial, "species": e["species"], "subfamily": e["subfamily"], "code": e["code"],
                                       "colony": e["colony"], "body_um3": e["body_um3"]}
    t0 = time.time()
    v, nlow = lowres_stack(vial, cfg["lowres_zstep"])
    if v is None: rec["status"] = "no_lowres"; return rec
    try:
        V, body, brain, info = LS.seg(v, e["body_um3"])
    except ValueError as ex:
        rec["status"] = "no_brain_proposal"; rec["error"] = str(ex)[:200]; return rec
    zz, yy, xx = np.nonzero(brain); off = info["crop_offset"]
    # reduced coords -> low-res stack coords: z unchanged (already zstep-subsampled), y/x x2, plus crop offset
    lz = (zz + off[0]) * cfg["lowres_zstep"]; ly = (yy + off[1]) * 2; lx = (xx + off[2]) * 2
    url = f"{ROOT}/{vial}.tif"; info_t = BT.layout(url); size, npages = BT.n_pages(url, info_t)
    fz, fy, fx = npages / nlow, info_t["H"] / v.shape[1], info_t["W"] / v.shape[2]
    lo = np.array([lz.min() * fz, ly.min() * fy, lx.min() * fx]); hi = np.array([(lz.max() + 1) * fz, (ly.max() + 1) * fy, (lx.max() + 1) * fx])
    c, half = (lo + hi) / 2, (hi - lo) / 2 * (1 + cfg["pad_frac"])
    half = np.minimum(half, cfg["max_box_um"] / 2.44 / 2)
    z0, y0, x0 = np.maximum(np.floor(c - half), 0).astype(int)
    z1, y1, x1 = np.minimum(np.ceil(c + half), [npages, info_t["H"], info_t["W"]]).astype(int)
    if not BT.verify_offset(url, info_t, z1 - 1): rec["status"] = "layout_irregular"; return rec
    box = BT.crop(url, info_t, z0, z1, y0, y1, x0, x1)
    (out / "boxes").mkdir(exist_ok=True); (out / "qc").mkdir(exist_ok=True)
    p = out / "boxes" / f"{vial}.npy"; np.save(p, box); qc_png(box, out / "qc" / f"{vial}.png")
    rec.update(status="ok", url=url, file_bytes=size, n_pages=npages, lowres_slices=nlow, box_zyx=[int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
               box_shape=list(box.shape), box_sha256=hashlib.sha256(box.tobytes()).hexdigest(), lowres_brain_vox=int(brain.sum()),
               proposal=info.get("best"), sec=round(time.time() - t0, 1))
    return rec


def main(cfg, out):
    elig = json.loads(pathlib.Path("configs/eligible_workers.json").read_text(encoding="utf-8"))
    chosen = select(elig, cfg["n_specimens"], cfg["seed"])
    print(f"selected {len(chosen)} specimens over {len({e['subfamily'] for e in chosen})} subfamilies")
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf, cf.ThreadPoolExecutor(cfg["workers"]) as ex:
        for rec in ex.map(lambda e: one(e, cfg, out), chosen):
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n"); mf.flush()
            print(rec["vial"], rec["species"], rec["status"], rec.get("box_shape"), rec.get("sec"))
