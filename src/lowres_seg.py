# Adapted from agent F5 (SEL-FABLE-F5 ant_seg6.py): heuristic brain proposal on AntScan low-res stacks.
# Added: crop offset of the reduced volume so a proposal can be mapped back to stack coordinates.
# Brain measurement on low-res AntScan preview stacks. Version 6 (check A2) = version 4 with two changes:
# (1) body = union of all filled components >= 3 % of the largest (v4 kept only the largest, often just the head);
# (2) scale parameters and size limits use the owners' body volume when given.
# Body mask: per-slice bright-cuticle contour + 2D fill at full xy resolution, then 3D clean-up at half resolution.
# Brain: largest-scoring smooth soft-tissue blob surrounded by textured (muscle) tissue.
# usage: ant_seg4.py <vol.npy> <out_prefix> [zsub]
import sys, json, time
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

P = dict(tex_win=5, deep_c=0.03, open_c=0.02, fmin=0.002, fmax=0.12, nb_c=0.10, topk=8, rel_hi=0.6)


def otsu(x, bins=128):
    hb, eb = np.histogram(x, bins=bins); hb = hb.astype(float); c = (eb[:-1] + eb[1:]) / 2
    w0 = np.cumsum(hb); w1 = w0[-1] - w0; cs = np.cumsum(hb * c)
    m0 = cs / np.maximum(w0, 1); m1 = (cs[-1] - cs) / np.maximum(w1, 1)
    return float(c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))])


def body_threshold(v):
    sub = v[::3, ::2, ::2].astype(np.float32)
    f = ndi.gaussian_filter(sub, (0, 0.8, 0.8))
    hist, _ = np.histogram(f, bins=256, range=(0, 256))
    hs = ndi.uniform_filter1d(hist.astype(float), 5)
    mode = int(np.argmax(hs[20:]) + 20)
    r = mode
    while r < 255 and hs[r] > hs[mode] / 2:
        r += 1
    hw = max(r - mode, 3)
    start = mode + 2 * hw
    for i in range(start, min(start + 60, 235)):
        if hs[i] <= hs[i - 3:i + 4].min() + 1e-9 and hs[i + 4:i + 16].max() > 1.05 * hs[i]:
            return float(i), mode, hw
    return float(mode + 3.5 * hw), mode, hw


def reduce_stack(v, T):
    nz, H, W = v.shape
    H2, W2 = H // 2, W // 2
    V = np.zeros((nz, H2, W2), np.float32); S = np.zeros((nz, H2, W2), np.float32); B = np.zeros((nz, H2, W2), bool)
    w = P["tex_win"]
    for i in range(nz):
        a = v[i].astype(np.float32)
        g = ndi.gaussian_filter(a, 0.6)
        mu = ndi.uniform_filter(g, w)
        sd = np.sqrt(np.maximum(ndi.uniform_filter(g * g, w) - mu * mu, 0))
        br = ndi.gaussian_filter(a, 0.8) > T
        br = ndi.binary_closing(np.pad(br, 3), iterations=2)[3:-3, 3:-3]
        fl = ndi.binary_fill_holes(br)
        V[i] = a[: H2 * 2, : W2 * 2].reshape(H2, 2, W2, 2).mean((1, 3))
        S[i] = sd[: H2 * 2, : W2 * 2].reshape(H2, 2, W2, 2).mean((1, 3))
        B[i] = fl[: H2 * 2, : W2 * 2].reshape(H2, 2, W2, 2).max((1, 3))
    return V, S, B


def seg(v, body_um3=None):
    t0 = time.time()
    T, mode, hw = body_threshold(v)
    V, S, B = reduce_stack(v, T)
    B = ndi.binary_opening(B, iterations=1)
    lab, n = ndi.label(B)
    if n == 0:
        raise ValueError("no_body")
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= 0.03 * sizes.max()) + 1
    body = np.isin(lab, keep)
    zz_, yy_, xx_ = np.nonzero(body)
    mg = 14
    sl = tuple(slice(max(int(a.min()) - mg, 0), min(int(a.max()) + mg + 1, V.shape[i])) for i, a in enumerate((zz_, yy_, xx_)))
    off = [int(a.start) for a in sl]
    V = V[sl]; S = S[sl]; body = body[sl]
    del lab, B
    body = ndi.binary_closing(np.pad(body, 3), iterations=2)[3:-3, 3:-3, 3:-3]
    body = ndi.binary_fill_holes(body)
    nb_mask = int(body.sum())
    nb = float(body_um3 / 929.7) if body_um3 else float(nb_mask)
    cb = nb ** (1 / 3)
    mu = ndi.uniform_filter(V, 3); sd = ndi.uniform_filter(S, 3)
    dout = ndi.distance_transform_edt(~body)
    eth = (dout > 4) & (dout < 12)
    if eth.sum() < 500:
        eth = (dout > 2) & (dout < 12)
    sd_bg = float(np.median(sd[eth])); mu_bg = float(np.median(mu[eth]))
    del dout
    dist = ndi.distance_transform_edt(body)
    d0 = max(2.0, P["deep_c"] * cb); r0 = int(max(1, round(P["open_c"] * cb)))
    interior = dist > d0
    shell = body & (dist <= 2) & (V > T)
    cut = float(np.median(V[shell])) if shell.sum() > 100 else float(T + 30)
    scale = max(cut - mu_bg, 10.0)
    rel = (mu - mu_bg) / scale
    cand_vox = interior & (rel > -0.1) & (rel < P["rel_hi"])
    if cand_vox.sum() < 200:
        raise ValueError("few_interior_voxels")
    lsd = np.log(np.maximum(sd[cand_vox], 1e-3))
    thr_sd = float(np.exp(otsu(lsd)))
    thr_sd = float(np.clip(thr_sd, 2.0 * sd_bg, 8.0 * sd_bg))
    smooth = cand_vox & (sd < thr_sd)
    if smooth.sum() < 100:
        raise ValueError("few_smooth_voxels")
    rel_lo = float(np.clip(otsu(rel[smooth]), 0.04, 0.30))
    tissue = smooth & (rel > rel_lo)
    tissue = ndi.binary_opening(tissue, structure=ndi.generate_binary_structure(3, 1), iterations=r0)
    lab2, n2 = ndi.label(tissue)
    if n2 == 0:
        raise ValueError("no_tissue_component thr_sd=%.2f rel_lo=%.2f" % (thr_sd, rel_lo))
    sz = np.bincount(lab2.ravel())[1:]
    order = np.argsort(sz)[::-1][: P["topk"]]
    textured = interior & (sd > thr_sd) & (rel < 1.0)
    rN = int(max(3, round(P["nb_c"] * cb)))
    objs = ndi.find_objects(lab2)
    cands = []
    for o in order:
        s = objs[o]
        s2 = tuple(slice(max(a.start - rN, 0), min(a.stop + rN, lab2.shape[i])) for i, a in enumerate(s))
        comp = lab2[s2] == (o + 1)
        nbh = ndi.binary_dilation(comp, iterations=rN) & ~comp & interior[s2]
        score = float(textured[s2][nbh].mean()) if nbh.sum() > 30 else 0.0
        R = (3 * sz[o] / (4 * np.pi)) ** (1 / 3)
        shellvol = 4 / 3 * np.pi * ((R + rN) ** 3 - R ** 3)
        emb = float(textured[s2][nbh].sum() / shellvol)
        cz, cy, cx = [int(round(c)) for c in ndi.center_of_mass(lab2[s] == (o + 1))]
        cz += s[0].start; cy += s[1].start; cx += s[2].start
        Rb = int(round(P.get("ball_c", 0.30) * cb))
        bz = slice(max(cz - Rb, 0), min(cz + Rb + 1, lab2.shape[0])); by = slice(max(cy - Rb, 0), min(cy + Rb + 1, lab2.shape[1])); bx = slice(max(cx - Rb, 0), min(cx + Rb + 1, lab2.shape[2]))
        zz, yy, xx = np.ogrid[bz, by, bx]
        ball = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= Rb * Rb
        den = (interior[bz, by, bx] & ball).sum()
        musc = float((textured[bz, by, bx] & ball).sum() / max(den, 1))
        cands.append(dict(label=int(o + 1), vox=int(sz[o]), frac=round(float(sz[o] / nb), 5), score=round(score, 3), emb=round(emb, 3), musc=round(musc, 3),
                          rel=round(float(rel[s2][comp].mean()), 3)))
    ok = [c for c in cands if P["fmin"] <= c["frac"] <= P["fmax"] and c["musc"] >= P.get("musc_min", 0.40)]
    if not ok:
        raise ValueError("no_candidate_in_range " + json.dumps(cands[:4]))
    best = max(ok, key=lambda c: c["vox"])
    brain = lab2 == best["label"]
    brain = ndi.binary_closing(np.pad(brain, 2), iterations=1)[2:-2, 2:-2, 2:-2] & (dist > 1.5)
    brain = ndi.binary_fill_holes(brain)
    info = dict(T_body=T, mode=mode, hw=hw, sd_bg=round(sd_bg, 3), mu_bg=round(mu_bg, 2), cut=round(cut, 1), thr_sd=round(thr_sd, 2), rel_lo=round(rel_lo, 3),
                d0=round(float(d0), 2), r0=r0, rN=rN, body_vox=nb_mask, body_vox_ref=round(nb, 1), mask_ratio=round(nb_mask / nb, 3), n_parts=int(len(keep)), brain_vox=int(brain.sum()), brain_frac=round(float(brain.sum() / nb), 5),
                best=best, crop_offset=off, cands=cands[:5], sec=round(time.time() - t0, 1))
    return V, body, brain, info


def overlay(V, body, brain, out, r=44, scale=3):
    zc, yc, xc = [int(c) for c in ndi.center_of_mass(brain)]
    lo, hi = np.percentile(V[body], [2, 99])

    def panel(img, bd, br, c0, c1):
        a = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1) * 255
        rgb = np.stack([a, a, a], -1)
        rgb[bd & ~ndi.binary_erosion(bd)] = [40, 170, 255]
        rgb[br & ~ndi.binary_erosion(br)] = [255, 40, 40]
        q = np.zeros((2 * r, 2 * r, 3), np.uint8)
        a0, a1 = max(c0 - r, 0), min(c0 + r, rgb.shape[0]); b0, b1 = max(c1 - r, 0), min(c1 + r, rgb.shape[1])
        q[: a1 - a0, : b1 - b0] = rgb[a0:a1, b0:b1]
        return q

    row = np.concatenate([panel(V[zc], body[zc], brain[zc], yc, xc), panel(V[:, yc], body[:, yc], brain[:, yc], zc, xc)], 1)
    sil = body.any(2).astype(np.uint8) * 90
    loc = np.stack([sil, sil, sil], -1); loc[brain.any(2)] = [255, 40, 40]
    li = Image.fromarray(loc); h = 2 * r
    wd = max(1, min(int(li.size[0] * h / li.size[1]), 3 * r))
    row = np.concatenate([row, np.asarray(li.resize((wd, h)))], 1)
    Image.fromarray(row).resize((row.shape[1] * scale, row.shape[0] * scale), Image.LANCZOS).save(out)


if __name__ == "__main__":
    v = np.load(sys.argv[1])
    if len(sys.argv) > 3 and int(sys.argv[3]) > 1:
        v = v[:: int(sys.argv[3])]
    bu = float(sys.argv[4]) if len(sys.argv) > 4 else None
    try:
        V, body, brain, info = seg(v, bu)
    except ValueError as e:
        print(json.dumps(dict(fail=str(e)))); sys.exit(0)
    print(json.dumps(info))
    overlay(V, body, brain, sys.argv[2] + "_ov.png")
