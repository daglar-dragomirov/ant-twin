import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
z = np.load("C:/Users/Daglar/AppData/Local/Temp/ant_twin_fig_cache/tk063_img_lab.npz"); img, lab = z["img"], z["lab"]
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for j, k in enumerate([100, 160, 220]):
    for i, (L, t) in enumerate([(lab[k], "their label, as stored"), (np.roll(lab[k], (110, 28), (0, 1)), "their label shifted down 110, right 28 px")]):
        a = ax[i, j]; a.imshow(img[k], cmap="gray"); e = L & ~ndi.binary_erosion(L, iterations=2)
        a.imshow(np.ma.masked_where(~e, e), cmap="autumn", alpha=1); a.set_title(f"slice {k}: {t}"); a.axis("off")
plt.tight_layout(); plt.savefig("C:/Users/Daglar/AppData/Local/Temp/claude/ec4acf495cf9ca86/7768b5bd-aa1d-4e1a-9bf1-c0c2aac1d3cd/scratchpad/tk063_check.png", dpi=70)
