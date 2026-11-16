"""Specimen-level data for the preregistered measurement-error model (prereg sec. 6).
Specimens = exactly those averaged into the 270 primary species means (main_ok, body present, the species' chosen stain).
Known measurement SD m (log10 units), seg v3 held-out relative brain-volume error, median |error| -> SD via /0.6745:
  m_main = log10(1 + 0.05/0.6745) = 0.0310  (AntScan held-out, median |error| ~5%; folds 3.0-8.9%)
  m_tk   = log10(1 + 0.02/0.6745) = 0.0127  (Toulkeridou human labels, median |error| ~2%; folds 1.0-2.6%)
Between-fold spread (prereg) is not available: fold_cv was not written by the seg v3 ensemble prediction."""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import *
import numpy as np

s = pd.read_csv(RUN_OUT / "specimens_table.csv", dtype={"vox_class": str}, low_memory=False)
p = primary_data()
x = s[s.main_ok & s.log10_body.notna() & s.species_key.isin(p.species_key)].merge(
    p[["species_key", "stain", "tip", "log10_colony_any", "log10_brain"]].rename(columns={"log10_brain": "sp_mean"}), on="species_key")
x = x[x.stained == x.stain].copy()
chk = x.groupby("species_key").log10_brain.mean().reindex(p.species_key)
assert len(chk) == 270 and np.allclose(chk.to_numpy(), p.log10_brain.to_numpy()), "specimens do not reproduce species means"
m_main, m_tk = np.log10(1 + 0.05 / 0.6745), np.log10(1 + 0.02 / 0.6745)
out = pd.DataFrame({"species": x.tip.to_numpy(), "species_key": x.species_key.to_numpy(), "log10_brain": x.log10_brain.to_numpy(),
                    "log10_body": x.log10_body.to_numpy(), "log10_colony": x.log10_colony_any.to_numpy(),
                    "iodine": (x.stained == "iodine").astype(int).to_numpy(), "vox122": (x.vox_class == "1.22").astype(int).to_numpy(),
                    "vox611": (x.vox_class == "6.11").astype(int).to_numpy(), "m_main": m_main, "m_tk": m_tk})
out.to_csv(WORK / "me_data.csv", index=False)
print(len(out), out.species.nunique(), round(m_main, 4), round(m_tk, 4), out[["iodine", "vox122", "vox611"]].sum().to_dict())
