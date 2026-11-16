import numpy as np
z = np.load("C:/Users/Daglar/AppData/Local/Temp/ant_twin_fig_cache/tk063_img_lab.npz"); lab = z["lab"]
pred = np.load("C:/Users/Daglar/AppData/Local/Temp/ant_twin_fig_cache/tk_pred_TK_063.npz")["pred"] > 0
def shift(a, b):
    cc = np.fft.ifft2(np.fft.fft2(a.astype(float)) * np.conj(np.fft.fft2(b.astype(float)))).real; dy, dx = np.unravel_index(cc.argmax(), cc.shape)
    return (dy - 520 if dy > 260 else dy), (dx - 520 if dx > 260 else dx)
D = lambda a, b: 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
for k in range(10, 309, 30):
    dy, dx = shift(pred[k], lab[k]); print(k, "shift", (dy, dx), "Dice raw", round(D(pred[k], lab[k]), 3), "after", round(D(pred[k], np.roll(lab[k], (dy, dx), (0, 1))), 3), "areas", int(pred[k].sum()), int(lab[k].sum()))
