# Stage files and their origin

Every experiment ran on its own branch; `main` collects the latest version of each file. To repeat a specific run, check out the branch recorded in that run's `meta.json`.

| File | Purpose | Taken from branch | Commit |
|---|---|---|---|
| `src/analysis/__init__.py` |  | `orx/brain-analysis-v1-blind-final-table-haller-gate` | `f252954` |
| `src/analysis/owners_cuticle_replication.R` |  | `orx/brain-analysis-v1-blind-final-table-haller-gate` | `f252954` |
| `src/analysis/pgls.py` |  | `orx/brain-analysis-v1-blind-final-table-haller-gate` | `f252954` |
| `src/analysis/phylolm_check.R` |  | `orx/brain-analysis-v1-blind-final-table-haller-gate` | `e97f905` |
| `src/bigtiff_range.py` |  | `orx/body-volume-from-low-res-cache-calibrated-agains` | `f66147f` |
| `src/lowres_seg.py` |  | `orx/body-volume-from-low-res-cache-calibrated-agains` | `7ff5125` |
| `src/r2.py` |  | `orx/baseline-run-contract-smoke-test` | `4b09dfb` |
| `src/runlog.py` |  | `orx/body-volume-from-low-res-cache-calibrated-agains` | `a153cfb` |
| `src/stages/__init__.py` |  | `orx/baseline-run-contract-smoke-test` | `4b09dfb` |
| `src/stages/body_volume.py` | body volume from the downsampled stacks (v8) | `orx/body-volume-v8-v6-with-true-page-counts-for-z-sp` | `5a26d3c` |
| `src/stages/boxes_grow.py` | enlarge boxes where the brain touched the box edge | `orx/grown-boxes-round-2-grow-again-where-the-grown-m` | `56f7c23` |
| `src/stages/boxes_v3.py` | full-resolution head boxes by HTTP range reads | `orx/boxes-v5-alt-b-one-download-per-file-all-boxes-c` | `d53e136` |
| `src/stages/boxes_v4.py` | head boxes re-cut with true slice counts | `orx/boxes-v4b-same-re-cut-from-the-other-end-second` | `7fc0f92` |
| `src/stages/brain_analysis.py` | preregistered PGLS analysis, gates and robustness checks | `orx/brain-analysis-v1-blind-final-table-haller-gate` | `e97f905` |
| `src/stages/brain_boxes.py` | first full-resolution head boxes (training specimens) | `orx/brain-boxes-v2-150-training-specimens-from-all-f` | `a4d81d2` |
| `src/stages/brain_freeze.py` | frozen brain table with checksum | `orx/brain-freeze-v1-final-brain-table-seg-v3-ensembl` | `6c477f6` |
| `src/stages/cuticle_fullres.py` | first own cuticle pipeline (superseded by cuticle_matte) | `orx/cuticle-full-res-v1-body-cuticle-and-soft-tissue` | `f245d4a` |
| `src/stages/cuticle_matte.py` | body and cuticle with the owners' program (Matte et al. 2025) | `orx/cuticle-matte-v2e-slices-of-a-specimen-processed` | `2a87e74` |
| `src/stages/cuticle_replication.py` | replication of the owners' colony–cuticle analysis | `orx/brain-analysis-v1-blind-final-table-haller-gate` | `f252954` |
| `src/stages/finder_alt.py` | boxes from the second and third finder candidates | `orx/finder-alt-boxes-from-2nd-3rd-finder-candidates` | `d1c59e9` |
| `src/stages/finder_data_v2.py` | training data for the 3D head finder (nnU-Net) | `orx/finder-dataset-v5c-crop-each-case-to-the-specime` | `ecf64e8` |
| `src/stages/finder_fast.py` | fast finder variant used for alternative candidates | `orx/finder-alt-boxes-from-2nd-3rd-finder-candidates` | `7fd1a4e` |
| `src/stages/finder_predict.py` | head finder inference on all stacks | `orx/finder-alt-boxes-from-2nd-3rd-finder-candidates` | `cf3ac5b` |
| `src/stages/lowres_cache.py` | downsampled stacks used by the head finder and body volume | `orx/body-volume-from-low-res-cache-calibrated-agains` | `9899d20` |
| `src/stages/lowres_finder_data.py` | training data for the head finder (first version) | `orx/body-volume-from-low-res-cache-calibrated-agains` | `185b514` |
| `src/stages/nni_drafts.py` | interactive-segmentation drafts used to bootstrap labels | `orx/nninteractive-drafts-v3-on-150-v2-boxes-voxel-aw` | `baa70dd` |
| `src/stages/nnunet_predict_scaled.py` | scale-normalised transfer test of the label-trained model | `orx/body-volume-from-low-res-cache-calibrated-agains` | `eee5ad4` |
| `src/stages/nnunet_tk.py` | nnU-Net training on the Toulkeridou et al. (2023) labels | `orx/finder-v3-final-skip-giant-cases-150-mvox-in-tra` | `5f743a5` |
| `src/stages/review_pack.py` | review sheets for the vision-agent quality control | `orx/body-volume-from-low-res-cache-calibrated-agains` | `25295bc` |
| `src/stages/seg_predict.py` | brain segmentation of every head box (five-fold ensemble) | `orx/seg-final-alt-ensemble-on-secondary-candidate-an` | `9d4b04a` |
| `src/stages/seg_v1.py` | training of the brain segmentation model (seg v3, five folds) | `orx/seg-v3-5-fold-nnu-net-2d-on-tk-all-agent-approve` | `c4c54d4` |
| `src/stages/smoke.py` | run-contract smoke test | `orx/baseline-run-contract-smoke-test` | `4b09dfb` |
| `src/stages/specimen_table.py` | AntScan workers joined to colony size and the phylogeny | `orx/brain-boxes-v2-150-training-specimens-from-all-f` | `a4d81d2` |
| `src/stages/tiny_ends.py` | boxes for very small ants (both specimen ends) | `orx/tiny-ant-path-both-specimen-ends-as-full-res-box` | `92d1db5` |
| `src/stages/tk_audit.py` | audit of the reference labels (re-prediction of all held-out cases) | `orx/tk-reference-audit-re-predict-all-117-held-out-t` | `7a85296` |
| `src/stages/tk_test_eval.py` | independent test on the Toulkeridou testing archive | `orx/tk-independent-test-v2-mask-polarity-both-naming` | `3210f79` |
| `src/stages/tk_volumes.py` | conversion of the Toulkeridou training archive to nnU-Net format | `orx/body-volume-from-low-res-cache-calibrated-agains` | `59a8d8e` |
| `src/stages/true_pages.py` | true slice count of every full-resolution stack (from the first TIFF directory) | `orx/true-page-counts-from-ifd0-for-all-antscan-stack` | `b76bb45` |
| `src/vendor/__init__.py` |  | `orx/cuticle-matte-gap-1-control-every-slice-authors` | `3c12467` |
| `src/vendor/matte2025/README.md` |  | `orx/cuticle-matte-gap-1-control-every-slice-authors` | `3c12467` |
| `src/vendor/matte2025/__init__.py` |  | `orx/cuticle-matte-gap-1-control-every-slice-authors` | `3c12467` |
| `src/vendor/matte2025/internal__auto_scan_segmentation.py` |  | `orx/cuticle-matte-gap-1-control-every-slice-authors` | `3c12467` |
| `src/vendor/matte2025/internal__extract_data_segmentation.py` |  | `orx/cuticle-matte-gap-1-control-every-slice-authors` | `3c12467` |
| `src/analysis/robustness_extra.py` | helpers for checks R2 and R8 (Borowiec tree, Bell-Roberts flags) | analysis worktree | — |
