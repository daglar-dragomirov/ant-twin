import sys, io, re, zipfile, numpy as np, tifffile
sys.path.insert(0, r"C:\projects\ant-twin\src"); import r2
exec(open("C:/Users/Daglar/AppData/Local/Temp/s3f.py").read())
zf = zipfile.ZipFile(io.BufferedReader(S3F(c, B, "ant-twin/data/toulkeridou2023/training.zip"), 1 << 20))
def load(stack):
    names = {}
    for n in zf.namelist():
        m = re.match(rf"^training/{stack}_(-?\d+)(_mask)?\.tif$", n)
        if m: names.setdefault(int(m.group(1)), [None, None])[1 if m.group(2) else 0] = n
    idx = sorted(k for k, v in names.items() if v[0] and v[1])
    msk = np.stack([tifffile.imread(io.BytesIO(zf.read(names[k][1]))) for k in idx]) > 127
    img = np.stack([tifffile.imread(io.BytesIO(zf.read(names[k][0]))) for k in idx])
    return img, msk
img, lab = load("Parasyscia_crib_rinodis2")
pred = np.load(r"C:\Users\Daglar\AppData\Local\Temp\ant_twin_fig_cache\tk_pred_TK_063.npz")["pred"] > 0
print("shapes", img.shape, lab.shape, pred.shape, "label frac", lab.mean().round(3), "pred frac", pred.mean().round(3))
D = lambda a, b: 2 * (a & b).sum() / (a.sum() + b.sum())
print("identity", round(D(pred, lab), 4))
for name, t in [("flip z", lab[::-1]), ("flip y", lab[:, ::-1]), ("flip x", lab[:, :, ::-1]), ("transpose yx", lab.transpose(0, 2, 1)),
                ("rot90", np.rot90(lab, 1, (1, 2))), ("rot270", np.rot90(lab, 3, (1, 2))), ("rot180", np.rot90(lab, 2, (1, 2)))]:
    print(name, round(D(pred, t), 4))
best = max(((dz, round(D(pred[max(0, dz):len(pred) + min(0, dz)], lab[max(0, -dz):len(lab) - max(0, dz)]), 4)) for dz in range(-40, 41)), key=lambda x: x[1])
print("best z-shift", best)
# in-plane shift by FFT cross-correlation of summed masks
A = pred.sum(0).astype(float); Bm = lab.sum(0).astype(float)
cc = np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(Bm))).real; dy, dx = np.unravel_index(cc.argmax(), cc.shape)
dy = dy - 520 if dy > 260 else dy; dx = dx - 520 if dx > 260 else dx
print("in-plane shift (dy,dx)", dy, dx, "Dice after shift", round(D(pred, np.roll(lab, (dy, dx), (1, 2))), 4))
np.savez_compressed(r"C:\Users\Daglar\AppData\Local\Temp\ant_twin_fig_cache\tk063_img_lab.npz", img=img, lab=lab)
