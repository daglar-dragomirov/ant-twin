# ANT-TWIN — final brain-volume table v1 (15 Nov 2026)

**Coverage.** 689 colony-worker specimens with an agent-approved brain mask, 325 of 352 species with known colony size (92%). The full target set is 818 specimens / 352 species. Table: `brain_table_final_v1.csv`; R2 copy: `ant-twin/meta/brain_table_final_v1.csv`.

**Pipeline (final round)**
1. True page counts were read from IFD0 for all 811 stacks. This fixed the size-based estimate, which was wrong for 551 deflate-compressed stacks.
2. Boxes were re-cut with the true z (boxes_v4), regrown for brains cut off by the border (twice), and a finder was retrained on 497 approved low-res brains (isotropic 12 µm, specimen-cropped).
3. Final finder pass over the 288 specimens still without a brain, plus secondary finder candidates for the failures.
4. Seg v3 was trained as a 5-fold nnU-Net 2d model on Toulkeridou 2023 + 530 agent-approved AntScan brains. Its 5-fold ensemble (no TTA) is the final uniform segmentation for every specimen.
5. Every mask was checked visually by vision agents (Opus 5.5, high effort). For the 530 previously approved brains, the ensemble masks agree with the approved masks: Dice median 0.988 (p10 0.972), volume ratio median 0.996 (IQR 0.990–1.001), 0 disagreements.

**Accuracy of seg v3 on held-out species (5 folds)**
- Toulkeridou (human labels): Dice 0.962–0.973, |volume error| 1.0–2.6%.
- Gate-1 AntScan drafts: Dice 0.917–0.956, |volume error| 3.0–8.9%. Small n per fold.

**Within-species repeatability.** 175 species with ≥2 workers. ICC(1) of log brain volume = 0.90; median within-species CV = 13% (IQR 6.7–26%; this includes true worker polymorphism).

**Sources of the 689 masks**
| Source | Masks |
|---|---|
| Re-cut boxes | 462 |
| Grown boxes | 63 + 5 + 7 |
| Final finder | 125 |
| Secondary candidates | 27 |

**Not found (27 species / 129 specimens).** Mostly scans with almost no soft-tissue contrast, damaged or empty specimens, and tiny ants where no candidate box held the head. 7 stacks are missing from the low-res cache.
