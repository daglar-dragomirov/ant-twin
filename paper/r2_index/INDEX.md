# ANT-TWIN — index of everything stored in Cloudflare R2

Bucket `dsc-biology`, prefix `ant-twin/`. All paths below are relative to `ant-twin/`.
Snapshot of R2 taken **2026-11-15 ~18:10 UTC**: 15,680 objects, 267.5 GB (GB = 10^9 bytes), not counting the three index files.
Companion files: `runs/RUNS.csv` and `runs/RUNS.md` (one row per run folder). Local copies: dsc-biology repo `research/ant-twin/r2_index/`.

## 1. Project in one paragraph

ANT-TWIN tests whether worker brain volume scales with colony size across ~350 ant species. Data: AntScan synchrotron micro-CT, 820 colony-worker specimens of 352 species with known colony size.
Pipeline:
1. Specimen table and a low-res copy of every scan.
2. A low-res "finder" (nnU-Net 3d) locates the brain, and a full-res head box is cut around it.
3. Brain segmentation: final model seg v3, a 5-fold nnU-Net 2d ensemble.
4. Every mask is checked by vision agents (QC).
5. Frozen brain table: 689 specimens / 325 species, `brains.jsonl` SHA-256 `df1902…6ead`.
6. Body volume (v8, with owners' S1 where available).
7. Preregistered PGLS: primary test, robustness R1–R14, R2 Borowiec tree, R8, measurement-error model.

Result: β₂ = +0.035 [0.017; 0.054], support H2. Workers in larger colonies have relatively larger brains.
Latest stage (done 16 Nov): own cuticle / soft-tissue measurement with the authors' Matte 2025 segmentation, extending check R3 to 235 species.

## 2. How to read this index

- **Run folders.** `runs/<YYYYMMDD-HHMMSS>_<orx-branch-slug>_<host>/`.
  - The time is the UTC start. The host is a RunPod container id or `DESKTOP-7Q7S0N3` (the local PC).
  - A timestamp identifies a folder uniquely. The one exception is `20261114-193224`, which has two folders. So this index often names runs as `runs/<timestamp>_*`.
- **Inside a run folder:**
  - `meta.json`: stage, git branch and commit, orx run id, orx title and parent, host, start/end, status, `r2_verified`;
  - `stdout.log` and `env.txt`;
  - `outputs/`: results, and often `summary.json`;
  - sometimes `inputs/`: copies of the inputs, for the analysis runs.
- **Caveat.** When a run was cancelled, its `meta.json` was never rewritten. It still says `status: running`, `r2_verified: false`, even when the outputs were uploaded completely (by sync or by hand). `RUNS.csv` therefore also gives the orx outcome.
- **Status legend** (column `status` in RUNS.csv):
  - accepted: outputs used by a later step or reported;
  - superseded: finished, but replaced by a later version;
  - cancelled: stopped by us;
  - failed: crashed;
  - in progress: still running at the snapshot;
  - unknown.
  A run can be accepted although orx shows it as cancelled or failed, for example the seg-final passes that were cancelled after all masks existed.
- **Feeds legend:**
  - **FREEZE**: direct input of the frozen brain table;
  - **PRIMARY**: direct input of the primary analysis;
  - **ROBUST**: input of the robustness checks;
  - **PREREG**: fixed N/hash or power for the preregistration;
  - **UPSTREAM**: indirect input (labels, models, boxes, QC approvals that led to the frozen table);
  - **VALID**: accuracy or replication check;
  - **R3X**: cuticle / soft-tissue extension of R3 (done).
- **Specimen keys.** A key is `<folder>__<vial>`, e.g. `5x__26-48` or `GAGA_10x__GAGA-01-37`. Suffixes: `@A`/`@B` are the two ends of a tiny ant; `@C1`/`@C2` are the 2nd and 3rd finder candidates. Cache objects are `<key>.npz`.
- **Fetching files.** Use the helper `C:/projects/ant-twin/src/r2.py`: `python -I r2.py ls dsc-biology ant-twin/runs/20261115-093934`, or `r2.client()` (boto3). Credentials are read at runtime from the owner's credential file and are not stored anywhere in R2 or git.

## 3. Top-level prefixes

| Prefix | What it holds | Objects | Size |
|---|---|---|---|
| `runs/` | One folder per run (153 folders), plus `RUNS.csv` / `RUNS.md` | 9,501 | 77.8 GB |
| `cache/` | Shared derived data written by runs: low-res stacks, full-res brain boxes, brain masks (table 3a) | 5,770 | 146.9 GB |
| `checkpoints/` | nnU-Net checkpoints and training logs synced from pods every 10 min (crash insurance, `scripts/ckpt_sync.py`), plus copies of trained weights pulled onto pods for inference (table 3b) | 230 | 27.4 GB |
| `data/` | Mirrored external data: Toulkeridou 2023 (Dryad), Economo 2018 Dryad zip, Matte 2025 owners' datasets and R scripts | 54 | 14.1 GB |
| `meta/` | Reference files: key lists, item lists, true page counts, vision-agent QC verdicts, phylogenies, final brain table (section 5) | 81 | 1.08 GB |
| `analysis/` | `robustness_v1/`: locally computed R8, R2 (Borowiec), measurement-error model; `r3_extension_v1/`: final R3 extension | 42 | 0.12 GB |
| `review/` | vision-agent mask-review sheets per round (brains_v1…v5, alt, grown, gate1): QC images + verdict files; mirror of dsc-biology research/ant-twin/review/ (uploaded 16 Nov) | 1818 | — |
| `RUNLOG.jsonl` | Append-only start/end log of every run (262 lines, 153 runs; matches `runs/`) | 1 | 0.08 MB |
| `README.md` | First layout note (13 Nov, Russian). **Outdated**: it lists `data-manifests/`, `labels/`, `models/`, `results/`, `figures/` and `preprint/`, which were never created | 1 | <0.01 MB |
| `INDEX.md` | This file | 1 | — |

There is no `review/` prefix in R2. QC verdicts are in `meta/*_auto_review/`, QC images in `runs/<seg run>/outputs/qc/`, and batches and prompts in git (`research/ant-twin/review/<wave>/`).

### 3a. `cache/`

| Prefix | Objects | Size | Content (producing node) | Role |
|---|---|---|---|---|
| `cache/lowres/` | 811 | 90.3 GB | Low-res stacks of 811/818 colony workers (`946f4b26`) | Input of finder and body volume |
| `cache/boxes_v3/` | 807 | 22.9 GB | Full-res boxes from round-1 finder detections (`5da8a096` rounds 1–4, tiny-ant ends) | Superseded: z is wrong for deflate-compressed stacks |
| `cache/boxes_v4/` | 820 | 23.5 GB | Boxes re-cut with true page counts (`efc51512` + `028e83f1`): 562 re-cut + 258 copied from v3 | UPSTREAM: 462 of them are in the freeze |
| `cache/boxes_grow/` | 78 | 1.9 GB | Grown boxes for brains cut by a box face (`6b3a27ea` + `1fe51fec`) | UPSTREAM: 63 in the freeze |
| `cache/boxes_grow2/` | 8 | 0.22 GB | Grow round 2 (`48eea837`) | UPSTREAM: 5 in the freeze |
| `cache/boxes_grow3/` | 10 | 0.25 GB | Grown v5 for final-finder boxes (`884168ba`) | UPSTREAM (alt group) |
| `cache/boxes_v5/` | 447 | 7.5 GB | 268 boxes for finder v8/v8b detections (`fc39b1e8` + `31a346d7`), plus 179 secondary-candidate boxes `@C1/@C2` (`6b19266e` + `ec652ae7`) | UPSTREAM |
| `cache/brain_masks_v1/` | 110 | 12 MB | Seg v1 masks (`0b9f07ca`) | QC wave brains_v1 |
| `cache/brain_masks_v2/` | 786 | 96 MB | Seg v2 masks on boxes_v3 (`4fcf9d71`) | QC waves v2/v3 |
| `cache/brain_masks_v4/` | 820 | 105 MB | Seg v2 masks on boxes_v4 (`a6025516`; unchanged keys copied from v2) | Reference masks for seg final; QC v4a–d |
| `cache/brain_masks_grow/`, `cache/brain_masks_grow2/` | 78 / 8 | 13 MB | Seg v2 masks on grown boxes (`60fcf822`, `d90d0a41`) | QC grown / grown2 |
| **`cache/brain_masks_final/`** | 798 | 98 MB | **Seg v3 ensemble masks**: 530 items of `meta/seg_final_items.json` + 268 of `seg_final_items_v5.json` | **FREEZE** |
| **`cache/brain_masks_final_alt/`** | 189 | 9 MB | **Seg v3 ensemble masks** on alt + grown-v5 boxes (`seg_final_items_alt.json`) | **FREEZE** |

Not every final mask is in the table. The freeze keeps the 689 approved specimens (section 4, stage 10).

### 3b. `checkpoints/` and where the model weights live

`checkpoints/<pod>/res/<DatasetNNN>/…/fold_k/` holds training checkpoints (best, latest and final) and logs, synced during training. `res_seg/`, `res_fast/`, `res_finder/`, `res_scaled/` and `res_test/` are copies of finished weights downloaded onto a pod for inference; they duplicate the canonical files below.

| Model | nnU-Net dataset | Canonical weights | Role |
|---|---|---|---|
| TK nnU-Net 2d fold 0 | Dataset501 (Toulkeridou) | `runs/20261113-190155_*/outputs/checkpoint_final.pth` | Bootstrap (gate 1) |
| Finder v1, 3d_fullres f0 | Dataset502 (48 cases) | `runs/20261114-000250_*/outputs/checkpoint_final.pth` | Round-1 finder |
| Seg v1, 2d, 5 folds | Dataset503 | `runs/20261114-002744_*/outputs/` (f0), `runs/20261114-025017_*/outputs/fold_1,2/`, `runs/20261114-025249_*/outputs/fold_3,4/` | Masks v1 |
| Seg v2, 2d f0 | Dataset504 | `runs/20261114-091138_*/outputs/checkpoint_final.pth` | Masks v2, v4, grown |
| Finder v2, 5 folds | Dataset505 | Partial only: `checkpoints/pod3,pod6,pod9/res/Dataset505_AntBrainLowres2/` (best/latest, no final) | Cancelled (page-count bug) |
| **Finder final (v3), fold all** | Dataset508 (497 cases, iso 12 µm, cropped) | `runs/20261115-012204_*/outputs/checkpoint_final.pth` | **Final finder** (v8, v8b, alt) |
| **Seg v3, 2d, 5 folds** | Dataset509 (TK + gate-1 + 530 approved AntScan) | `runs/20261114-212341_*/outputs/fold_0,1,2/` + `runs/20261114-215207_*/outputs/fold_3,4/` | **Final segmentation** |

Dataset numbers 506 and 507 are finder datasets that were never trained successfully.

## 4. Pipeline map

For each stage, the accepted run(s) come first, then the other attempts. The full folder name and node of every run are in `runs/RUNS.md`.

### Stage 0 — run contract (infrastructure)
- **Accepted:** `runs/20261113-135722_run-metadata-from-orx-env_DESKTOP-7Q7S0N3/` (node `f80139b3`): meta.json from the ORX environment, folder named by node.
- Superseded: `20261113-135324` (main) and `20261113-135621` (nobranch). Both are pre-orx smoke tests with no orx run id.

### Stage 1 — specimen table
- **Accepted (PRIMARY):** `runs/20261113-153618_specimen-table-v2-fix-species-name-joins_DESKTOP-7Q7S0N3/` (node `26397421`): 820 colony-worker specimens, 352 species. `outputs/specimens.csv` and `species.csv` are inputs of the analysis.
- Superseded: `20261113-153420` (v1, `54c0d3ea`): the colony-size join failed (0 species with colony).

### Stage 2 — label bootstrap: Toulkeridou labels, gate 1 (UPSTREAM)
- **Accepted:**
  - `runs/20261113-181636_tk-volumes-v2-skip-macos-appledouble-entries_6945dce09bf2/` (`f32f0aad`): 117 TK stacks → Dataset501.
  - `runs/20261113-162908_brain-boxes-v2-150-training-specimens-from-all-f_6945dce09bf2/` (`0057169d`): 150 specimens → 85 full-res boxes.
  - `runs/20261113-181614_nninteractive-drafts-v3-on-150-v2-boxes-voxel-aw_6945dce09bf2/` (`af9a58dc`): 85 nnInteractive drafts (mask A).
  - `runs/20261113-190155_nnu-net-tk-transfer-test-v2-rerun-after-pod-disk_6945dce09bf2/` (`2ea0308a`): TK nnU-Net 2d f0, val Dice 0.970.
  - `runs/20261113-205550_scale-normalized-inference-tk-nnu-net-on-antscan_6945dce09bf2/` (`a070007d`): TK model on boxes rescaled to 520 px (mask B).
  - `runs/20261113-221824_gate-1-review-pack-nninteractive-v3-vs-scaled-tk_6945dce09bf2/` (`0c5f2366`): A-vs-B pack → `meta/gate1_auto_review/`, 24 good brains.
- Superseded:
  - `20261113-142934` (brain labels v1, 21 boxes);
  - `20261113-154639` and `20261113-160500`: nnInteractive drafts v1/v2 on those boxes.
- Failed:
  - `20261113-140219`, `20261113-155239`;
  - `20261113-180256`: AppleDouble files in the zip;
  - `20261113-184533`: pod disk full.
- Cancelled: `20261113-142046`.

### Stage 3 — low-res cache and true page counts
- **Accepted:**
  - `runs/20261114-003142_low-res-cache-fetch-all-820-colony-worker-low-re_43cd6c057240/` (`946f4b26`): 811/818 stacks → `cache/lowres/` (90.3 GB); 6 have no low-res stack, 1 error.
  - `runs/20261114-154115_true-page-counts-from-ifd0-for-all-antscan-stack_a419f2c0b82a/` (`69f4119f`): true page count of all 811 stacks → `meta/true_pages.json`. The page estimate was wrong for 551 deflate-compressed stacks. This bug invalidated boxes_v3 z, finder dataset v2 and body z spacing, and everything downstream was redone.

### Stage 4 — body volume
- **Accepted (PRIMARY):** `runs/20261114-160633_body-volume-v8-v6-with-true-page-counts-for-z-sp_5d5eb7649d55/` (`51a4bc76`): 811 bodies; against S1, n = 278, median ratio 1.036. Rule used by the analysis: S1 > v8-valid > missing.
- Superseded:
  - v1–v5 (`20261114-010532`, `-011325`, `-012342`, `-014008`, `-020215`): tube included, or body dropped;
  - v6 (`20261114-024355`, and the full-cache rerun `20261114-060927`, used only by the dry run);
  - v7 (`20261114-040748`): does not fix bright-medium scans.
- Note in git: `results/body_volume.md` (written for v6/v7).

### Stage 5 — brain finder (low-res)
- **Round 1 (finder v1 model; UPSTREAM of the 530 brains that came through boxes_v3 → boxes_v4).**
  - Model: `runs/20261114-000250_*` (`972f6d35`) on Dataset502 (`runs/20261113-234553_*`, `9a60713b`).
  - Inference rows were merged by key from:
    - `20261114-052105` (v2, 145 rows, crashed on RAM);
    - `-060616` (v2 tail, 192 rows);
    - `-072011` (v3, 17);
    - `-080647` (v4, 16);
    - `-083611` (v5, 112);
    - `-103301` (**v7c, 427 rows: 398 ok**);
    - `-105047` (v7d, 3);
    - `-113510` (v7e, 1);
    - `-115210` (v7f, 46);
    - `-131215` (v7h, 20/20);
    - `-115731` (v7g small-ant rescue, 3 of 26);
    - `-121922` (tiny-ant ends, 46 rows).
  - Without useful rows:
    - cancelled `-045935` (8× TTA), `-080652`, `-083609`, `-092159`, `-095011`, `-102849`;
    - failed `-054646`, `-054930`, `-095015`, `-101936`, `-103302`.
- **Finder retraining (several dead ends).**
  - Finder v2: dataset `20261114-124449` (superseded); 5 folds `-130304`, `-130325`, `-131407`, `-131408`, `-134110`, all cancelled because of the page-count bug.
  - Datasets v3 `-160631` and v4 `-185956`: superseded.
  - Finder v3 folds: 12 runs `-192945` … `-200642`, 11 failed and 1 cancelled (dataset still uploading, then concurrent preprocessing / OOM on anisotropic data).
  - v3b folds `-202254`, `-202256`, `-202259`: cancelled or failed.
  - Dataset v5 `-203734`: cancelled.
  - Dataset v5b `-210725`: superseded.
  - Final-finder attempts `-220428` and `20261115-003240`: cancelled, too slow.
- **Accepted final finder:**
  - Dataset `runs/20261114-225259_finder-dataset-v5c-crop-each-case-to-the-specime_668c59782304/` (`d76aeeb0`): Dataset508, 497 cases.
  - Model `runs/20261115-012204_finder-v3-final-skip-giant-cases-150-mvox-in-tra_668c59782304/` (`3ae3d228`): fold all, 471 cases. orx shows it cancelled during the meaningless fold-all validation; the checkpoint was uploaded by hand (`MANUAL_UPLOAD.txt`).
- **Final finder pass** over the 288 specimens without an approved brain (`meta/finder_v8_targets.json`):
  - `runs/20261115-030654_finder-v8-true-page-counts-for-z-spacing-and-box_668c59782304/` (`2aeafcf8`): 214 rows, then CUDA OOM.
  - `runs/20261115-042152_finder-v8b-memory-light-argmax-resume-remaining_668c59782304/` (`b9e6e73e`): remaining 74.
  - Together: 268 ok, 13 no_brain, 7 not cached.
  - Secondary candidates: `runs/20261115-075129_finder-alt-boxes-from-2nd-3rd-finder-candidates_668c59782304/` (`4c58356e`), 179 rows for 99 of 111 failures.

### Stage 6 — full-res brain boxes
- **Boxes v3** (superseded by v4 for compressed stacks, still UPSTREAM through the copied keys), node `5da8a096` and children:
  - round 1 `20261114-063434`;
  - round 2 `-093448`;
  - round 3 `-115530`;
  - round 3b `-140643` and `-150654`;
  - round 4 `-135459` and `-145229`;
  - tiny-ant ends `-123035`.
- **Boxes v4 (accepted):**
  - `runs/20261114-160627_boxes-v4-re-cut-compressed-stack-boxes-with-true_a419f2c0b82a/` (`efc51512`, forward) and `runs/20261114-160709_boxes-v4b-same-re-cut-from-the-other-end-second_5d5eb7649d55/` (`028e83f1`, reverse) → `cache/boxes_v4/` (820).
  - Both were cancelled when the two passes met; partial indexes are uploaded.
- **Grown boxes (accepted):**
  - `20261114-165422` (v2, 10 grown);
  - `runs/20261114-182214_grown-boxes-v3-add-v4a-v4b-reviews-all-re-cut-ke_5d5eb7649d55/` (`1fe51fec`, 78 → `cache/boxes_grow/`);
  - `runs/20261114-195820_grown-boxes-round-2-grow-again-where-the-grown-m_5d5eb7649d55/` (`48eea837`, 8 → `boxes_grow2/`);
  - `runs/20261115-075318_grown-boxes-v5-cut-off-brains-in-the-final-finde_668c59782304/` (`884168ba`, 10 → `boxes_grow3/`).
  - Failed: `20261114-165222` (import error).
- **Boxes v5 (accepted):**
  - `runs/20261115-035116_boxes-v5-full-res-boxes-for-finder-v8-detections_668c59782304/` (`fc39b1e8`) + `runs/20261115-045206_boxes-v5b-v8-v8b-detections-reverse-order-second_5d5eb7649d55/` (`31a346d7`) → 268 boxes.
  - Alt: `runs/20261115-080804_boxes-v5-alt-b-one-download-per-file-all-boxes-c_668c59782304/` (`6b19266e`) + `runs/20261115-080951_boxes-v5-alt-b-shard-1-2_c742fe5c40be/` (`ec652ae7`) → 179 `@C1/@C2` boxes.
  - Cancelled: `20261115-075313` (alt v1, too slow).

### Stage 7 — segmentation models
- Seg v1, UPSTREAM: `20261114-002744` (`5b24a027`, f0; TK Dice 0.975, AntScan n = 5 Dice 0.88), `-025017` (folds 1+2), `-025249` (folds 3+4).
- Seg v2, UPSTREAM: `runs/20261114-091138_seg-v2-add-64-agent-approved-seg-v1-brains-self_9b718f84cbc4/` (`591756a7`): AntScan n = 7 Dice 0.91, |vol err| 6.3%. Failed first try: `20261114-083032`.
- **Seg v3 (final model):** `runs/20261114-212341_seg-v3-5-fold-nnu-net-2d-on-tk-all-agent-approve_9f0551f91543/` (`f0a0e364`, folds 0–2) and `runs/20261114-215207_seg-v3-folds-3-4_5d5eb7649d55/` (`95e50d2b`, folds 3–4). Held-out per fold: TK Dice 0.962–0.973, |vol err| 1.0–2.6%; metrics in `outputs/fold_k/summary.json` and `val_per_case.json`. Failed first try of folds 3–4: `20261114-212344` (pod disk full).

### Stage 8 — intermediate mask rounds and vision-agent QC (UPSTREAM: they decided which boxes were approved)

| Masks | Runs | QC verdicts |
|---|---|---|
| v1 (110) | `20261114-072018` (`0b9f07ca`) | `meta/brains_v1_auto_review/` |
| v2 (786) | `20261114-113518`, `-141504` (partial), `-145230` (`4fcf9d71`) | `meta/brains_v2_auto_review/`, `brains_v3_auto_review/` |
| v4 (820) | `20261114-163623`, `-172115`, `-180615`, `-183636` (`a6025516`) | `meta/brains_v4a…v4d_auto_review/` |
| grown (78), grown2 (8) | `20261114-185636`, `-193630` (`60fcf822`); `-203757` (`d90d0a41`) | `meta/brains_grown_auto_review/`, `brains_grown2_auto_review/` |

The approved boxes form `meta/seg_final_items.json`: 530 = boxes_v4 462 + grow 63 + grow2 5.

### Stage 9 — final segmentation: seg v3 5-fold ensemble, no TTA (FREEZE)

| Item list | Runs (all listed in the freeze config) | Masks → |
|---|---|---|
| `meta/seg_final_items.json` (530) | `runs/20261115-043851_seg-final-seg-v3-5-fold-ensemble-on-the-best-box_5d5eb7649d55/` (`710ff833`), `runs/20261115-045203_seg-final-b-same-ensemble-items-in-reverse-order_668c59782304/` and `runs/20261115-055335_seg-final-b-same-ensemble-items-in-reverse-order_668c59782304/` (`17ce7b4c`) | `cache/brain_masks_final/` |
| `meta/seg_final_items_v5.json` (268) | `runs/20261115-065118_seg-final-v5-ensemble-on-boxes-for-finder-v8-det_5d5eb7649d55/` (`1033b56d`), `runs/20261115-065120_seg-final-v5-reverse-second-gpu_668c59782304/` (`162aad99`); QC `meta/brains_v5_auto_review/` | `cache/brain_masks_final/` |
| `meta/seg_final_items_alt.json` (189) | `runs/20261115-083646_seg-final-alt-ensemble-on-secondary-candidate-an_c742fe5c40be/` (`5c4a2a6d`), `runs/20261115-085137_seg-final-alt-reverse_668c59782304/` (`468dbf61`); QC `meta/brains_alt_auto_review/` | `cache/brain_masks_final_alt/` |

- Run pairs work through the item list from both ends and were cancelled once all masks existed. `outputs/brains_at_cancel.jsonl` is the complete per-run record.
- `045203` crashed when a second seg job started on the same pod, but its 180 masks are used.
- Cancelled: `20261115-055129` (no outputs).
- For the 530 previously approved brains, ensemble vs approved mask: Dice median 0.988, 0 disagreements (`results/brain_table_v1.md`).

### Stage 10 — brain freeze (feeds PRIMARY)
- **`runs/20261115-093934_brain-freeze-v1-final-brain-table-seg-v3-ensembl_DESKTOP-7Q7S0N3/`** (`f7337bb6`):
  - `outputs/brains.jsonl` contains 689 specimens, SHA-256 `df1902432f4680b295c1e68cc68804c6d42a71a445d0dbd2003f90bbaf646ead`;
  - `outputs/review_final.json`;
  - sources: boxes_v4 462, grow 63, grow2 5, finder v8 125, alt / grown v5 34.
- The table as CSV: `meta/brain_table_final_v1.csv` (325/352 species; within-species ICC 0.90).

### Stage 11 — analysis
- Dry run, superseded: `20261114-121035` (`60f76779`, 28 species).
- **Blind run (PREREG):** `runs/20261115-094208_brain-analysis-v1-blind-final-table-haller-gate_DESKTOP-7Q7S0N3/` (`c9a8f45e`): Haller gate passed (n = 270, slope 0.489), synthetic recovery OK. N and hash went into prereg v1.0.
- **Primary (PRIMARY, ROBUST):** `runs/20261115-100746_brain-analysis-v1-primary-unblinded-preregistere_DESKTOP-7Q7S0N3/` (`73e0da9c`):
  - β₂ = +0.035 [0.017; 0.054], p = 0.0002, n = 270, λ 0.72 → support H2;
  - `outputs/primary.json` holds the R1–R14 results; `outputs/sources.json` gives the R2 path and SHA-256 of every input (specimen table v2, S1, body v8, freeze);
  - `inputs/` holds copies of the inputs.
- **Robustness (ROBUST):** `analysis/robustness_v1/`, computed locally from the primary outputs, not an orx run:
  - R8 (`r8.json`): +0.040 [0.020; 0.061], n = 251;
  - R2, Borowiec 2025 tree (`r2*.json`, `r2_reps_*.csv`): median +0.029, pooled [0.001; 0.058], 117 genera;
  - measurement-error model (`me/brms_*`, `me/mcmcglmm_*`, `me_data.csv`): +0.038 [0.019; 0.055];
  - reproduction of the primary fit: `repro.py`, `run_outputs/`.

### Stage 12 — cuticle / soft tissue, extension of R3 (R3X) — **DONE (16 Nov)**
- **Result:** dsc-biology `research/ant-twin/results/r3_extension_v1.md` (copy: `analysis/r3_extension_v1/r3_extension_v1.md`). Soft-tissue β₂ +0.010 [−0.006; 0.026], 235 species (154 S1 + 81 ours), undecided; strict QC rule +0.002 [−0.015; 0.019], equivalence; body, same species +0.034 [0.014; 0.055].
- **Combined input / outputs:** `analysis/r3_extension_v1/` — `cuticle_final.jsonl` (185 specimens, all thresholds), `gap1_vs_gap3.csv`, `r3x.json`.
- **Accepted runs** (owners' Matte 2025 program, vendored unchanged; every 3rd slice by HTTP range; each run has `outputs/cuticle.jsonl`, `slices/<key>.npz`, `qc/<key>.jpg`):
  - Matte v2 (`40e82cf9`, shards `51ecbe4c`, `ae8e5d31`): `runs/20261115-164942_*`, `-165210_*`, `-170953_*` (9 ok rows, cancelled for v2b);
  - Matte v2b (`264d642e`): `runs/20261115-172355_*` — all compressed keys (142 ok);
  - Matte v2c (`c5cf7acc`): `runs/20261115-180617_*` — 9 ok uncompressed;
  - Matte v2d (`60a953e7`): `runs/20261115-195412_*` — 18 ok uncompressed;
  - Matte v2e (`b333c996`): `runs/20261115-222211_*` — last 7 uncompressed (chunked mode), done + verified;
  - gap-1 control (`41bd798a`): `runs/20261115-180820_*` — 8 validation specimens at every slice; gap-3/gap-1 = 0.9996.
- **Keys:** `meta/cuticle_v2_keys.json` + `cuticle_v2_selection.json` (185 = 141 targets + 44 S1 validation); helpers `cuticle_v2_done.json`, `cuticle_v2c_keys.json`, `cuticle_v2c_done.json`, `cuticle_v2e_keys.json`, `cuticle_gap1_keys.json`.
- **Code:** ant-twin branches `orx/cuticle-matte-*`; vendored program `src/vendor/matte2025/` (source zip also at `data/owners_matte2025/python_scripts/Python_scripts.zip`). Analysis script: dsc-biology `research/ant-twin/scripts/robustness_v1/r3x_run.py`.
- Earlier attempts:
  - cancelled `20261115-151715` (own thresholds leak on bright-medium scans);
  - cancelled `-155123` (Matte pilot: full-stack downloads too slow);
  - cancelled `-162507` (v1 full shard 0/5, 0 rows);
  - failed `-165232`, `-165244` (local Windows backend).

### Side checks (VALID)
- `runs/20261113-233327_tk-independent-test-v2-mask-polarity-both-naming_6945dce09bf2/` (`c5e6927c`): TK model on unseen testing.zip species. Final numbers in `results/tk_independent_test.md`: Dice median 0.965, |vol err| 1.3% on 17 species. Superseded invalid v1: `20261113-231635`.
- `runs/20261113-181741_cuticle-replication-owners-colony-cuticle-pgls-w_DESKTOP-7Q7S0N3/` (`70183aef`): the owners' colony→cuticle PGLS, replicated at −0.041/SD as published; power table for the prereg. Earlier run of the same node: `20261113-181033`.

## 5. Key reference files in `meta/`

| File / prefix | Content |
|---|---|
| `meta/brain_table_final_v1.csv` | Final brain-volume table v1 (689 specimens / 325 species); same as git `results/brain_table_final_v1.csv` |
| `meta/true_pages.json` | 811 stacks: true page count from IFD0, compression, size-based estimate and ratio |
| `meta/finder_v8_targets.json` | 288 specimens without an approved brain after round 1 → finder v8 |
| `meta/finder_alt_targets.json` | 111 specimens that failed after finder v8 → 2nd/3rd candidates |
| `meta/seg_final_items.json`, `_v5.json`, `_alt.json` | Item lists `[key, box prefix, reference-mask prefix]` for the seg-final runs (530 / 268 / 189) |
| `meta/gate1_auto_review/`, `meta/brains_{v1,v2,v3,v4a,v4b,v4c,v4d,grown,grown2,v5,alt}_auto_review/` | Vision-agent QC verdicts per wave (`auto_review_N.json`, some `auto_review_summary.json`) |
| `meta/economo2018/15k_all_ant_trees/` | Economo 2018 MCC + posterior trees (NCuniform stem = primary tree; FBD = R1) + `MANIFEST.json` (SHA-256) |
| `meta/borowiec2025/` | Borowiec 2025 (Zenodo 14338504): `10-MCMCTree.zip` (855 MB), extracted chronograms, consensus tree, `README.txt`, `manifest.json` (R2 check) |
| `meta/cuticle_pilot_keys.json` | 33 S1 specimens for the cuticle pilot |
| `meta/cuticle_test1/2/3.json` | 1 / 3 / 1 debug keys for cuticle runs |
| `meta/cuticle_all_keys.json` | 689 freeze specimens (Matte v1 full shards; not used further) |
| `meta/cuticle_v2_keys.json`, `cuticle_v2_selection.json` | 185 v2 keys = 141 targets (species without S1 soft tissue) + 44 S1 validation, with the selection rule |
| `meta/cuticle_v2_done.json` | 9 keys finished by the cancelled v2 shards |
| `meta/cuticle_v2c_keys.json` | 34 uncompressed-stack keys for v2c |

Also in `data/`:
- `data/toulkeridou2023/training.zip` (9.32 GB) and `testing.zip` (4.62 GB): Dryad 10.5061/dryad.qz612jmgv;
- `data/economo2018/doi_10_5061_dryad_g579t7k__v20190404.zip`;
- `data/owners_matte2025/`: datasets S1–S7 (S1 = individual body/cuticle measurements, used by the analysis), R scripts, Sci. Adv. supplement, and our replication outputs (`replication_out/`).

## 6. Git: write-ups, preregistration, code

- **Write-ups:** repo `github.com/daglar-dragomirov/dsc-biology`, folder `research/ant-twin/results/`.
  - `brain_table_v1.md` + `brain_table_final_v1.csv`: commit `dfe1b00`;
  - `primary_v1.md`: `2b06793`;
  - `robustness_v1.md`: `3ca645e`;
  - earlier notes: `seg_v1.md` (`6168bc6`), `tk_independent_test.md` (`15f3e8e`), `body_volume.md` (`24c157e`, `55bc8a1`), `brains_v1_review.md` (`2644388`).
- **Preregistration:** `research/ant-twin/prereg/`.
  - `prereg_v1.0_frozen.md` was frozen in commit `3b1e350` before the primary test. Tag **`ant-twin-prereg-v1.0`** points to `c01e2ee` (wording fix).
  - `prereg_draft.md` (`d4b2a7d`) holds the full plan, sec. 6–8.
- **QC batches and prompts:** `research/ant-twin/review/<wave>/`. The QC jpgs are gitignored; they are in R2 under `runs/<seg run>/outputs/qc/`.
- **Code:** repo `github.com/daglar-dragomirov/ant-twin` (local `C:/projects/ant-twin`).
  - Every orx node is a git branch `orx/<slug>`. The slug is the middle part of the run-folder name.
  - The stage code is `src/stages/<stage>.py` and the configuration is `configs/experiment.json` on that branch. The exact commit of each run is in its `meta.json`.
  - orx project id: `28888eac-f570-4762-ad55-0da3ae96faf4` (`orx project view …` lists all 141 nodes).
- **Robustness scripts:** dsc-biology `research/ant-twin/scripts/robustness_v1/`; copies of `r8.py`, `repro.py` and `fetch.py` are in `analysis/robustness_v1/`.

## 7. Caveats for readers

- `meta.json` of 41 finished runs still says `running` / `r2_verified: false` (cancelled runs; see section 2). Check `orx_outcome` in RUNS.csv.
- Stale orx states:
  - seg v3 run `77459771` (`20261114-212341`) still shows `running`, although `meta.json` says done and verified;
  - `9de070c0` (`20261115-162507`) and `bf8497cd` (no R2 folder) show `Cancelling`.
- Large superseded data that is kept on purpose:
  - `cache/boxes_v3/`: about a third of it is duplicated in `boxes_v4/` by design;
  - full-size `.npy` predictions in the early TK-transfer runs: `20261113-190155`, `-205550`, `-221824`, together ≈ 17 GB;
  - nnU-Net raw datasets in the finder-dataset runs: v4 7.7 GB, v5 (cancelled, partial) 2.9 GB, v5b 14.1 GB, v5c 10.2 GB.
- `summary.json` of the blind and primary runs labels the model-based body source as `v6`. The actual input is the body v8 run (see `outputs/sources.json`).
