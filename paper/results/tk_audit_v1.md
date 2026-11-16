# TK reference-label audit, v1 (16 Nov 2026)

Run: orx node `4fa55ae4` (child of seg v3 `f0a0e364`); R2 `ant-twin/runs/20261116-061344_tk-reference-audit-re-predict-all-117-held-out-t_fbb50b957159/` (done, r2_verified). Code: `ant-twin` `src/stages/tk_audit.py`, commit `7a85296`.

**Why.** The lowest held-out case (TK_063, *Parasyscia cribrinodis*, Dice 0.495) looked misaligned in Fig. S2. A check on the deposited files alone (no model) showed that its label in the Dryad `training.zip` is rigidly offset from the image (about +110…+117 px in y, +20…+37 px in x on every slice); shifted back, it outlines the brain (`preprint/figures/image_data/tk063_label_offset_check.png`). Our loader pairs image and mask slices by identical indices (1–310, same gap), so the offset is in the archive, not in our reading.

**What.** All 117 held-out TK cases were re-predicted in full with the seg v3 fold whose validation set contains them (as nnU-Net validation: 2d, mirroring, tile step 0.5, Gaussian).

For each case the label was tested for a rigid offset:
- global in-plane shift by FFT cross-correlation of the z-projections;
- per-slice shifts (median and IQR);
- best z shift within ±20 slices.

**Flag rule.** The rule was fixed in the config before the run: shift ≥ 5 px, Dice gain ≥ 0.10 and per-slice IQR ≤ 15 px.

**Results.**
- **Reproduction:** the re-predictions reproduce the stored per-case Dice, with a maximum difference of 0.0001.
- **Flagged:** only **TK_063**. Its shift is (115, 27) px and the per-slice IQR is 14 px. The Dice rises from 0.495 to 0.812 after the shift. The label also covers about 1.4× the predicted area.
- **Median Dice:** 0.9686 over all 117 cases and 0.9687 without TK_063, so it is unchanged.
- **Other low cases are not offsets:** TK_083 *Pogonomyrmex pima* (0.859) and TK_032 *Dorylus kohli* (0.881) show no consistent shift (gain < 0). They are ordinary boundary or size disagreements.

**Use in the paper.** One neutral sentence in the Fig. S2 legend. The case stays in all statistics. No retraining is needed: TK_063 was 1 of about 650 training cases in 4 of the 5 folds, and the model outlines its brain correctly.

**Lesson (memory rule).** External reference data is audited at intake: overlays and offset/flip/scale checks.
