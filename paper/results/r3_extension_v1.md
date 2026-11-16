# ANT-TWIN — R3 extension: soft-tissue denominator for 235 species, v1 (16 Nov 2026)

Declared after the primary run; not preregistered (prereg sec. 11 deviation log applies). Extends the preregistered R3 (S1 soft tissue, 154 species, `primary_v1.md`).

## Measurement

**Program.** Body and cuticle volumes come from the owners' own segmentation program (Matte et al. 2025, Zenodo 15546461, CC BY 4.0).
- The program is vendored unchanged in `ant-twin` `src/vendor/matte2025/`.
- The wrapper stage is `cuticle_matte`.
- The per-slice chain is the authors': `color_image_processing → remove_vial → segmente_image(threshold) → filter_segmented_image → fill_contours`.
- Per-specimen aggregation is also theirs: median filter over slices, then sum × voxel³.

**Specimens (185).**
- **141 targets:** primary-set specimens (main_ok, chosen stain, body present), at most 2 per species, from the 116 primary species without S1 soft tissue.
- **44 validation specimens:** a stratified sample of the S1 overlap.
- Selection is fixed in R2 `meta/cuticle_v2_selection.json`. All 185 were measured.

**Access.** Full downloads (~2.5 TB for all specimens) were not feasible, because the AntScan server throttles per IP.
- Only the slices actually used were read, via HTTP range requests: every 3rd slice, plus every 30th slice for the authors' z-limit search.
- The median filter of 25 slices became 9 samples.

**Gap-3 control.** The same program at every slice (the authors' setting) was run on 8 validation specimens. Ratio gap-3 / gap-1:

| Volume | Median | Range |
|---|---|---|
| Body | 0.9996 | 0.990–1.010 |
| Soft tissue | 0.9991 | 0.978–1.002 |

The subsampling is negligible (`gap1_vs_gap3.csv`).

**Threshold and QC rule.** The rule was fixed before the final fits and revised once on segmentation diagnostics; both versions are reported.
- Problem: with the authors' threshold 110, a bright mounting medium leaks into the mask. This happened in most 6.11 µm scans and some 2.44 µm scans. In these scans thresholds 130 and 145 agree with each other.
- The authors' scripts contain both 110 and 145.
- **Main ("adaptive") rule.** Per specimen:
  - use threshold 110 if body(110)/body(130) is within 1/1.10–1.10;
  - otherwise use 145 if body(145)/body(130) is within that range;
  - otherwise exclude the specimen;
  - in all cases, require cuticle/body between 0.03 and 0.40.
- **Sensitivity ("strict") rule:** threshold 110 only, with body stable across all three thresholds within 1.15.
- **Disclosure:** an interim R3x estimate on 28 added species (+0.007) had been seen before the rule was written. The revision to the adaptive rule was made without looking at R3 estimates under it.

**Pass rates.**
- Adaptive rule: 141 of 185 specimens pass (110: 92; 145: 49).
- Failures by voxel size: 6.11 µm 17/31, 2.44 µm 26/145, 1.22 µm 1/9.

## Validation against S1 (owners' values for the same specimens)

| Rule (n) | Body ratio | Cuticle ratio | Soft ratio | r(log body) | r(log soft) |
|---|---|---|---|---|---|
| Adaptive (41 of 43) | 0.998 | 1.04 | 0.992 | 0.998 | 0.997 |
| Threshold 110, stable specimens (37) | 0.998 | 1.06 | 0.992 | 0.997 | 0.997 |

The adaptive rule excluded exactly the three validation specimens that disagree with S1:
- 2x__M-9 and 2x__S-15: 6.11 µm scans with a bright medium;
- 5x__12-41: body 1.77 × S1, and the same at every slice.

On the 41 accepted specimens our soft tissue reproduces S1 to within 1% (median), with r = 0.997.

## Result (primary model with soft tissue replacing body; PGLS-λ, Economo MCC)

| Denominator / set | β₂ (log10 colony) | 95% CI | n species | Outcome (sec. 7) |
|---|---|---|---|---|
| Soft tissue, S1 only (preregistered R3) | +0.006 | [−0.013; +0.024] | 154 | undecided |
| **Soft tissue, S1 + ours, adaptive rule (main)** | **+0.010** | [−0.006; +0.026], p = 0.23 | 235 (154 S1 + 81 ours) | undecided |
| Same, 100 posterior trees (pooled) | median +0.010 | [−0.006; +0.026], no tree p < 0.05 | 235 | undecided |
| Body, the same 235 species | +0.034 | [+0.014; +0.055], p = 0.001 | 235 | support H2 |
| Soft tissue, strict rule (sensitivity) | +0.002 | [−0.015; +0.019] | 197 | equivalence (H0) |
| Body, the same 197 species | +0.027 | [+0.007; +0.047] | 197 | support H2 |

- corr(log10 colony, log10 soft/body) = +0.13 over the 235 species.
- λ is estimated at 0 in the soft-tissue fits.

## Reading

1. The soft-tissue result now rests on 235 of the 270 primary species instead of 154.
2. With soft tissue as the size measure, the colony coefficient stays near zero under both rules:
   - main rule: +0.010, CI excludes negative values beyond −0.006;
   - strict rule: +0.002, within the SESOI of ±0.02, i.e. formally equivalent to no effect.
3. On the same species with total body as the size measure, it is +0.027 to +0.034 (support H2).
4. **Conclusion:** relative to soft tissue, worker brain investment does not change with colony size. The positive association relative to total body size comes from relatively thinner cuticle in species with larger colonies (Matte et al. 2025).
5. **H1 is still rejected:** no estimate in any set is negative beyond −0.015.

## Files

- **Stage code:** `ant-twin` branches `orx/cuticle-matte-v2*` (nodes 40e82cf9, 264d642e, c5cf7acc, 60a953e7, b333c996) and `orx/cuticle-matte-gap-1-control-*` (41bd798a).
- **Runs (R2 `ant-twin/runs/`):** `20261115-164942/165210/170953_*v2*`, `20261115-172355_*v2b*`, `20261115-180617_*v2c*`, `20261115-195412_*v2d*`, `20261115-222211_*v2e*` (done, verified), and the gap-1 control `20261115-180820_*`. Each run holds `cuticle.jsonl`, `slices/<key>.npz` (per-slice pixel counts per threshold) and `qc/<key>.jpg`.
- **Combined input and outputs:** R2 `ant-twin/analysis/r3_extension_v1/`:
  - `cuticle_final.jsonl` (185 specimens);
  - `gap1_vs_gap3.csv`;
  - `r3x.json`.
- **Script:** `research/ant-twin/scripts/robustness_v1/r3x_run.py`. QC rule commits: `0d198ce` (strict), `cc9a07e` (adaptive revision).
- **Engineering fixes on the way** (none affects the per-slice program):
  - keep-alive range reader;
  - numpy `readinto` bug (uncompressed stacks);
  - uncompressed files carry one page more than `meta/true_pages.json`, so a tolerance of 1 page was added;
  - chunked slice processing for CPU-bound scans, verified identical to unchunked.

## Correction (16 Nov 2026, after the preprint draft checks)

The "Reading" above compares soft tissue (same instrument) with body from the primary table (S1 or our v8 mask). Like-for-like, with body and soft tissue from the same instrument on the same specimens (S1 or the owners' program), the colony coefficient is: 235 species — body +0.014 [−0.004; 0.031], soft tissue +0.010; 154 S1 species — body +0.012, soft +0.006. The cuticle accounts for only ~0.005; most of the gap to +0.034 comes from the body-volume source (our v8 body mask underestimates some bodies: 6 workers with brain > 10% of body). Points 3–4 of the Reading are withdrawn. What stands: no estimate is below −0.015 (brains do not shrink, H1 rejected); a positive effect is not supported like-for-like. Script: `research/ant-twin/preprint/tables/supp_checks.py` (independently reproduced). Planned for the journal version: re-measure all v8-body specimens with the owners' program (67 left + 44 QC failures with `vial_effort`).
