"""Build RUNS.csv / RUNS.md for ant-twin/runs/ from merged.json (meta.json + orx) and a manual status table.
status: accepted | superseded | cancelled | failed | in progress | unknown
feeds : FREEZE (direct input of brain_freeze) | PRIMARY (direct input of primary analysis) | ROBUST | PREREG |
        UPSTREAM (indirect: labels, models, boxes, approvals leading to the frozen table) | VALID (accuracy check) |
        R3X (cuticle / soft-tissue extension, in progress) | INFRA | -
"""
import csv, json, sys, pathlib

S = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
merged = json.load(open(S / "merged.json", encoding="utf-8"))
orx = json.load(open(S / "orx_runs.json", encoding="utf-8"))

A, SU, C, F, IP = "accepted", "superseded", "cancelled", "failed", "in progress"
# key: run_dir timestamp (+ node prefix where the timestamp is shared)
T = {
 "20261113-135324": (SU, "INFRA", "pre-orx smoke test of the run contract on main (dirty tree); no science output"),
 "20261113-135621": (SU, "INFRA", "smoke test without a branch; no science output"),
 "20261113-135722": (A, "INFRA", "run contract: meta.json from ORX env, R2 folder named by node"),
 "20261113-140219": (F, "-", "failed after 6 min (exit 1): 1 box written, manifest empty"),
 "20261113-142046": (C, "-", "4 rows, all error; cancelled"),
 "20261113-142934": (SU, "-", "40 stratified specimens -> 21 ok full-res boxes; path replaced by boxes v2 (150 specimens)"),
 "20261113-153420": (SU, "-", "colony-size join failed (0 worker species with colony); fixed in v2"),
 "20261113-153618": (A, "PRIMARY; UPSTREAM", "820 colony-worker specimens / 352 species; specimens.csv + species.csv are analysis inputs"),
 "20261113-154639": (SU, "-", "21 nnInteractive drafts on brain-labels-v1 boxes; replaced by drafts v3"),
 "20261113-155239": (F, "-", "failed after 29 s (exit 1): 1 mask written, drafts.jsonl empty"),
 "20261113-160500": (SU, "-", "21 drafts with smart clicks; replaced by drafts v3"),
 "20261113-162908": (A, "UPSTREAM", "150 specimens -> 85 ok full-res boxes (gate-1 set)"),
 "20261113-180256": (F, "-", "macOS AppleDouble ._* entries in training.zip; fixed in tk_volumes v2"),
 "20261113-181033": (SU, "-", "earlier local run of the cuticle-replication node (no orx run id); see 20261113-181741"),
 "20261113-181614": (A, "UPSTREAM", "85 nnInteractive drafts (mask A of the gate-1 review)"),
 "20261113-181636": (A, "UPSTREAM", "117 Toulkeridou 2023 stacks -> Dataset501 (TK part of all seg training sets)"),
 "20261113-181741": (A, "VALID; PREREG", "owners' colony->cuticle PGLS replicated (-0.041/SD as published; tree match 349/352); power table for prereg; local run, no orx run id"),
 "20261113-184533": (F, "-", "pod disk full (60 GB container disk)"),
 "20261113-190155": (A, "UPSTREAM", "TK nnU-Net 2d fold 0 (val Dice 0.970) + raw transfer to 85 AntScan boxes (weak); model reused by scale test"),
 "20261113-205550": (A, "UPSTREAM", "TK model on boxes rescaled to 520/700 px; 520 px better (45/85 dominant component); mask B of gate 1"),
 "20261113-221824": (A, "UPSTREAM", "gate-1 pack: 85 A-vs-B pairs (median Dice 0.13) -> vision-agent review meta/gate1_auto_review (24 good)"),
 "20261113-231635": (SU, "-", "invalid test (14 inverted masks, 2nd naming scheme, 2 training species); fixed in v2"),
 "20261113-233327": (A, "VALID", "TK model on unseen testing.zip species: Dice median 0.956 (23 stacks); final numbers in results/tk_independent_test.md"),
 "20261113-233536": (F, "-", "branch had the v1 lowres_stack; fixed in 9a60713b"),
 "20261113-234553": (A, "UPSTREAM", "Dataset502: 48 gate-1 brains mapped onto low-res stacks (finder v1 training set)"),
 "20261114-000250": (A, "UPSTREAM", "finder v1 model (3d_fullres fold 0, val Dice 0.38); used by all round-1 finder inference runs"),
 "20261114-002744": (A, "UPSTREAM", "seg v1 fold 0: TK Dice 0.975; AntScan n=5 Dice 0.88, |vol err| 12.7%; used by seg predict v1"),
 "20261114-003142": (A, "UPSTREAM; PRIMARY (via body v8)", "low-res cache: 811/818 stacks -> cache/lowres/ (90.3 GB); 6 without low-res, 1 error"),
 "20261114-010532": (SU, "-", "body v1: 89 error, 731 not cached (cache still filling)"),
 "20261114-011325": (SU, "-", "body v2: takes the whole tube (median ratio vs S1 x20)"),
 "20261114-012342": (SU, "-", "body v3: drops the body (median ratio vs S1 0.03)"),
 "20261114-014008": (SU, "-", "body v4: drops the body (median ratio vs S1 0.02)"),
 "20261114-020215": (SU, "-", "body v5: tube wall cut first; median ratio 1.14 but heavy upper tail"),
 "20261114-024355": (SU, "-", "body v6 on 340 cached specimens; rerun on the full cache in 20261114-060927"),
 "20261114-025017": (A, "UPSTREAM", "seg v1 CV folds 1+2 (AntScan Dice 0.88 / 0.87); folds used by seg predict v1"),
 "20261114-025249": (A, "UPSTREAM", "seg v1 CV folds 3+4 (AntScan Dice 0.88 / 0.92); fold 3 used by seg predict v1"),
 "20261114-040748": (SU, "-", "body v7 per-slice threshold: does not fix bright-medium scans"),
 "20261114-045935": (C, "-", "finder inference with 8x TTA: 1.3 min/specimen; 0 rows"),
 "20261114-052105": (A, "UPSTREAM", "partial: 145 finder rows (139 ok) before RAM crash; merged by key into boxes_v3"),
 "20261114-054646": (F, "-", "fresh pod3 without /root/nnunet dirs; 0 rows"),
 "20261114-054930": (F, "-", "retry on pod3 failed (0 rows); reason not recorded"),
 "20261114-060616": (A, "UPSTREAM", "partial: 192 finder rows (185 ok); cancelled when it met the pod1 run; merged into boxes_v3"),
 "20261114-060927": (SU, "-", "body v6 on the full cache (811); used only by the dry run; replaced by v8 (true page counts)"),
 "20261114-063434": (A, "UPSTREAM", "boxes_v3 round 1: 110 ok full-res boxes -> cache/boxes_v3"),
 "20261114-072011": (A, "UPSTREAM", "partial: 17 finder rows; cancelled (slow); merged in boxes round 2"),
 "20261114-072018": (A, "UPSTREAM", "seg predict v1: 110 masks -> cache/brain_masks_v1 -> QC wave brains_v1 (64 good)"),
 "20261114-080647": (A, "UPSTREAM", "partial: 16 finder rows; cancelled (70-90 s/specimen)"),
 "20261114-080652": (C, "-", "0 rows; cancelled (slow)"),
 "20261114-083032": (F, "-", "first seg v2 attempt failed (exit 1); retried as 20261114-091138"),
 "20261114-083609": (C, "-", "0 rows; pod1 memory pressure"),
 "20261114-083611": (A, "UPSTREAM", "partial: 112 finder rows; cancelled; merged in boxes round 2"),
 "20261114-091138": (A, "UPSTREAM", "seg v2 (Dataset504): AntScan n=7 Dice 0.91 / 6.3%, TK 0.975 / 1.5%; model for masks v2, v4, grown"),
 "20261114-092159": (C, "-", "0 rows; replaced by finder v6"),
 "20261114-093448": (A, "UPSTREAM", "boxes_v3 round 2 (all finder runs v1-v5): 194 ok, 110 cached, 6 error"),
 "20261114-095011": (C, "-", "finder v6: still ~96 s/specimen; 0 rows"),
 "20261114-095015": (F, "-", "finder v6 shard 0 died of RAM (pod4)"),
 "20261114-101936": (F, "-", "finder v7: OOM on upsampled 6.11 um scans after 5 rows"),
 "20261114-102849": (C, "-", "finder v7b: no outputs; replaced by the v7c split"),
 "20261114-103301": (A, "UPSTREAM", "finder v7c: 427 rows (398 ok, 26 no_brain, 3 not cached) -> boxes round 3/3b"),
 "20261114-103302": (F, "-", "finder v7c 6.11-um part: OOM (GPU shared with seg v2); 0 rows"),
 "20261114-105047": (A, "UPSTREAM", "partial: 3 rows; CPU path 19 min/scan -> cancelled; rows used in boxes round 4"),
 "20261114-113510": (A, "UPSTREAM", "partial: 1 row; 11 min/scan -> cancelled; row used in boxes round 4"),
 "20261114-113518": (A, "UPSTREAM", "seg predict v2: 201 masks -> cache/brain_masks_v2 -> QC wave brains_v2"),
 "20261114-115210": (A, "UPSTREAM", "partial: 46 rows (43 ok), then CUDA crash (>2^31 elements); rows used in boxes round 4"),
 "20261114-115530": (A, "UPSTREAM", "boxes_v3 round 3 (v7c detections): 392 ok, 6 error"),
 "20261114-115731": (A, "UPSTREAM", "finder v7g small-ant rescue at 2x scale: 3 of 26 found"),
 "20261114-121035": (SU, "-", "analysis machinery dry run (blind, 28 species, Haller pass); replaced by 20261115-094208 / 20261115-100746; local, no orx run id"),
 "20261114-121922": (A, "UPSTREAM", "tiny-ant path: 23 targets -> 46 end rows (key@A / key@B)"),
 "20261114-123035": (A, "UPSTREAM", "boxes for tiny-ant ends: 42 ok, 4 box_too_big"),
 "20261114-124449": (SU, "-", "finder v2 dataset (Dataset505, 154 cases): z labels wrong (page-count bug); training cancelled"),
 "20261114-130304": (C, "-", "finder v2 fold 0: cancelled (page-count bug); partial checkpoints only in checkpoints/pod3"),
 "20261114-130325": (C, "-", "finder v2 fold 1: cancelled (page-count bug); partial checkpoints in checkpoints/pod6"),
 "20261114-131215": (A, "UPSTREAM", "finder v7h: remaining 20 6.11-um scans, 20/20 ok (A100)"),
 "20261114-131407": (C, "-", "finder v2 fold 4: cancelled (page-count bug); partial checkpoints in checkpoints/pod3"),
 "20261114-131408": (C, "-", "finder v2 fold 3: cancelled (page-count bug); partial checkpoints in checkpoints/pod6"),
 "20261114-134110": (C, "-", "finder v2 fold 2: cancelled (page-count bug); partial checkpoints in checkpoints/pod9"),
 "20261114-135459": (A, "UPSTREAM", "boxes_v3 round 4 (v7d-v7h): 48 ok, 22 error (pod1 disk full)"),
 "20261114-140643": (A, "UPSTREAM", "boxes_v3 round 3b (reverse): 102 ok, 269 cached, 27 error"),
 "20261114-141504": (A, "UPSTREAM", "partial: 24 masks v2 on shared pod9; cancelled (slow), continued on pod10"),
 "20261114-145229": (A, "UPSTREAM", "boxes round 4 retry: 21 ok, 48 cached, 1 error"),
 "20261114-145230": (A, "UPSTREAM", "seg predict v2 (pod10): 561 masks (82 empty, 224 touch edge) -> QC waves"),
 "20261114-150654": (A, "UPSTREAM", "boxes round 3b retry: 0 new, 392 cached, 6 persistent errors"),
 "20261114-154115": (A, "UPSTREAM; PRIMARY (via body v8)", "true page counts for 811 stacks (551 compressed stacks had a wrong estimate) -> meta/true_pages.json"),
 "20261114-160627": (A, "UPSTREAM", "boxes_v4 forward pass (re-cut with true pages); cancelled when it met v4b; partial index uploaded"),
 "20261114-160631": (SU, "-", "finder dataset v3 (Dataset506, true pages, 154 cases); not trained, replaced by v4"),
 "20261114-160633": (A, "PRIMARY", "body v8 (v6 + true pages): 811 bodies; vs S1 n=278 median ratio 1.036; rule S1 > v8-valid"),
 "20261114-160709": (A, "UPSTREAM", "boxes_v4 reverse pass; cancelled when passes met; cache/boxes_v4 = 820"),
 "20261114-163623": (A, "UPSTREAM", "seg predict v4 pass 1: 167 masks -> QC brains_v4a"),
 "20261114-165222": (F, "-", "grown boxes v1: import error (fixed in v2)"),
 "20261114-165422": (A, "UPSTREAM", "partial: 10 grown boxes (126 no box yet); reused as cached by grown v3"),
 "20261114-172115": (A, "UPSTREAM", "seg predict v4 pass 2: 204 masks -> QC brains_v4b"),
 "20261114-180615": (A, "UPSTREAM", "seg predict v4 pass 3: 175 masks -> QC brains_v4c"),
 "20261114-182214": (A, "UPSTREAM", "grown v3: 68 grown + 10 cached -> cache/boxes_grow (78)"),
 "20261114-183636": (A, "UPSTREAM", "seg predict v4 pass 4: 59 masks -> QC brains_v4d; brain_masks_v4 complete (820)"),
 "20261114-185636": (A, "UPSTREAM", "seg on grown boxes, pass 1: 39 masks"),
 "20261114-185956": (SU, "-", "finder dataset v4 (Dataset507, 497 cases); anisotropic spacing -> training OOM; replaced by iso v5b/v5c"),
 "20261114-192945": (F, "-", "finder v3 fold 2: started before dataset upload finished (expected 497, found 308)"),
 "20261114-192946": (F, "-", "finder v3 fold 3: same (dataset not fully uploaded)"),
 "20261114-192949": (F, "-", "finder v3 fold 4: same (dataset not fully uploaded)"),
 "20261114-193224|058cb285": (F, "-", "finder v3 fold 0: same (dataset not fully uploaded)"),
 "20261114-193224|ad4150e4": (F, "-", "finder v3 fold 1: same (dataset not fully uploaded)"),
 "20261114-193630": (A, "UPSTREAM", "seg on grown boxes, rerun: 39 more masks (brain_masks_grow = 78)"),
 "20261114-195148": (F, "-", "finder v3 fold 0 relaunch: concurrent preprocessing -> blosc2 errors / OOM"),
 "20261114-195149": (C, "-", "finder v3 fold 1 relaunch: cancelled"),
 "20261114-195150": (F, "-", "finder v3 fold 2 relaunch: preprocessing failure"),
 "20261114-195151": (F, "-", "finder v3 fold 3 relaunch: preprocessing failure"),
 "20261114-195153": (F, "-", "finder v3 fold 4 relaunch: preprocessing failure"),
 "20261114-195820": (A, "UPSTREAM", "grown round 2: 8 regrown -> cache/boxes_grow2"),
 "20261114-200641": (F, "-", "finder v3 fold 2, 3rd try: preprocessing OOM (6.11-um cases upsampled)"),
 "20261114-200642": (F, "-", "finder v3 fold 4, 3rd try: preprocessing OOM"),
 "20261114-202254": (C, "-", "finder v3b fold 0 (np=4): anisotropic Dataset507 still too slow/OOM; abandoned"),
 "20261114-202256": (F, "-", "finder v3b fold 2: OOM (anisotropic Dataset507)"),
 "20261114-202259": (F, "-", "finder v3b fold 4: OOM (anisotropic Dataset507)"),
 "20261114-203734": (C, "-", "finder dataset v5 (scipy zoom) too slow (~40 s/case); partial 2.9 GB upload"),
 "20261114-203757": (A, "UPSTREAM", "seg on grown round-2 boxes: 8 masks -> cache/brain_masks_grow2"),
 "20261114-210725": (SU, "-", "finder dataset v5b (Dataset508, iso 12 um, uncropped); training preprocessing too slow; replaced by v5c"),
 "20261114-212341": (A, "UPSTREAM (model of seg final)", "SEG V3 folds 0-2 (Dataset509: TK + gate-1 + 530 approved AntScan); checkpoints in outputs/fold_k/; orx still shows this run as running (stale)"),
 "20261114-212344": (F, "-", "seg v3 folds 3-4, 1st try: pod9 disk full"),
 "20261114-215207": (A, "UPSTREAM (model of seg final)", "SEG V3 folds 3-4 checkpoints"),
 "20261114-220428": (C, "-", "finder final on uncropped v5b: preprocessing 86 s/case"),
 "20261114-225259": (A, "UPSTREAM", "finder dataset v5c: Dataset508, 497 cases, specimen-cropped, iso 12 um"),
 "20261115-003240": (C, "-", "finder final on v5c: giant cases (300-1150 Mvox) stalled preprocessing"),
 "20261115-012204": (A, "UPSTREAM", "FINDER FINAL (fold all, 471 cases, >150 Mvox dropped); cancelled in useless fold-all validation; checkpoint uploaded manually (MANUAL_UPLOAD.txt)"),
 "20261115-030654": (A, "UPSTREAM", "partial: finder v8 214/288 rows (205 ok) before CUDA OOM; rest done by v8b"),
 "20261115-035116": (A, "UPSTREAM", "boxes v5 pass 1: 121 boxes for v8 detections -> cache/boxes_v5"),
 "20261115-042152": (A, "UPSTREAM", "finder v8b: remaining 74 (63 ok, 10 no_brain, 1 not cached)"),
 "20261115-043851": (A, "FREEZE", "seg final (seg v3 5-fold ensemble) forward over 530 items; cancelled after all masks existed; brains_at_cancel.jsonl uploaded"),
 "20261115-045203": (A, "FREEZE", "seg final b (reverse): crashed when a 2nd seg job started on the same pod; 180 masks; listed in freeze base_runs"),
 "20261115-045206": (A, "UPSTREAM", "boxes v5b reverse (175 rows); cancelled as duplicate once boxes_v5 had all 268"),
 "20261115-055129": (C, "-", "seg final v5 launched on a busy pod (crashed seg final b); no outputs"),
 "20261115-055335": (A, "FREEZE", "seg final b relaunch (--force): 108 masks; cancelled when passes met"),
 "20261115-065118": (A, "FREEZE", "seg final v5 forward over 268 finder-v8 items; cancelled after completion"),
 "20261115-065120": (A, "FREEZE", "seg final v5 reverse; cancelled after completion"),
 "20261115-075129": (A, "UPSTREAM", "finder alt: 2nd/3rd candidates -> 179 rows for 99 of 111 failed specimens"),
 "20261115-075313": (C, "-", "boxes v5 alt: too slow (same file downloaded twice); 0 rows"),
 "20261115-075318": (A, "UPSTREAM", "grown v5: 10 grown -> cache/boxes_grow3"),
 "20261115-080804": (A, "UPSTREAM", "boxes v5 alt b shard 0/2: 91 secondary-candidate boxes (@C1/@C2)"),
 "20261115-080951": (A, "UPSTREAM", "boxes v5 alt b shard 1/2: 86 ok, 2 cached"),
 "20261115-083646": (A, "FREEZE", "seg final alt forward: 114 masks (47 empty); brains_at_cancel uploaded"),
 "20261115-085137": (A, "FREEZE", "seg final alt reverse: 129 masks"),
 "20261115-093934": (A, "PRIMARY", "BRAIN FREEZE v1: 689 specimens; brains.jsonl sha256 df1902...6ead; review_final.json"),
 "20261115-094208": (A, "PREREG", "blind run: Haller gate pass (n=270, slope 0.489), synthetic recovery ok; N + hash went into prereg v1.0"),
 "20261115-100746": (A, "PRIMARY; ROBUST", "PRIMARY: beta2 = +0.035 [0.017; 0.054], p = 0.0002, n = 270 -> support H2; R1-R14 in outputs"),
 "20261115-151715": (C, "-", "cuticle full-res v1 (own thresholds): leaks on bright-medium scans in pilot (7 rows)"),
 "20261115-155123": (C, "-", "Matte 2025 segmentation pilot (4 rows); full-stack downloads too slow"),
 "20261115-162507": (C, "-", "Matte v1 full shard 0/5: 0 rows; full downloads infeasible; orx stuck at 'Cancelling'"),
 "20261115-164942": (C, "R3X", "Matte v2 shard 0/3: 4 ok; cancelled for v2b; outputs uploaded manually, rows kept (meta/cuticle_v2_done.json)"),
 "20261115-165210": (C, "R3X", "Matte v2 shard 1/3: 3 ok, 1 error; cancelled for v2b; rows kept"),
 "20261115-165232": (F, "-", "Matte v2 shard 2/3 on local Windows backend: ProcessPool without __main__ guard"),
 "20261115-165244": (F, "-", "same local failure; second R2 folder for the same orx run 41c61d9b"),
 "20261115-170953": (C, "R3X", "Matte v2 shard 2/3 on pod13: 2 ok; cancelled for v2b; rows kept"),
 "20261115-172355": (IP, "R3X", "Matte v2b (keep-alive range reader) on pod13, remaining v2 keys; at snapshot 57 rows (50 ok, 7 error)"),
 "20261115-180617": (IP, "R3X", "Matte v2c (readinto fix) on 34 uncompressed keys (meta/cuticle_v2c_keys.json); no outputs at snapshot"),
}

rows = []
for x in merged:
    ts = x["run_dir"][:15]
    key = ts if ts in T else f"{ts}|{x['node'][:8]}"
    st, feeds, note = T[key]
    o = orx.get(x["orx_run"] or "", {})
    oc = o.get("status", "")
    if oc.startswith("Cancelling"): oc = "cancelling (stale)"
    rows.append({
        "run_dir": x["run_dir"],
        "date_utc": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}",
        "orx_node": x["node"] or "",
        "node_title": x["title"] or "",
        "stage": x["stage"] or "",
        "status": st,
        "feeds": feeds,
        "orx_run": (x["orx_run"] or "")[:8],
        "orx_outcome": oc.lower() if oc else "(no orx run)",
        "meta_status": x["meta_status"],
        "r2_verified": str(bool(x["r2v"])).lower(),
        "n_objects": x["n"],
        "size_MB": x["mb"],
        "note": note,
    })
missing = [k for k in T if not any((r["run_dir"][:15] == k) or (f"{r['run_dir'][:15]}|{r['orx_node'][:8]}" == k) for r in rows)]
assert not missing, missing
OUT.mkdir(parents=True, exist_ok=True)
cols = list(rows[0].keys())
with open(OUT / "RUNS.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

from collections import Counter
cnt = Counter(r["status"] for r in rows)
tot_n = sum(r["n_objects"] for r in rows); tot_mb = sum(r["size_MB"] for r in rows)

def esc(s): return str(s).replace("|", "\\|")
md = []
md.append("# ANT-TWIN — run folders in `ant-twin/runs/`\n")
md.append("One row per run folder (same content as `RUNS.csv`). Snapshot of R2 taken 2026-11-15 ~18:10 UTC. "
          "Entry point: `ant-twin/INDEX.md` (pipeline map, legends, reference files).\n")
md.append(f"**{len(rows)} run folders, {tot_n} objects, {tot_mb/1000:.1f} GB.** "
          + ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items(), key=lambda t: -t[1])) + ".\n")
md.append("**Columns.** `date_utc` comes from the folder name (UTC start time). `orx_node` is the experiment node "
          "(code = git branch `orx/<slug>` in github.com/daglar-dragomirov/ant-twin; exact commit in each `meta.json`). "
          "`status` is the role of the run in the project (legend in INDEX.md); `orx_outcome` is what orx recorded for the job. "
          "`meta_status` / `r2_verified` are copied from the run's `meta.json`: cancelled runs never rewrote it, so they still say "
          "`running` / `false` even when their outputs were uploaded and are complete.\n")
md.append("**feeds.** FREEZE = direct input of the frozen brain table; PRIMARY = direct input of the primary analysis; "
          "ROBUST = input of the robustness checks; PREREG = fixed N/hash or power for the preregistration; "
          "UPSTREAM = indirect input (labels, models, boxes, QC approvals leading to the frozen table); VALID = accuracy/replication check; "
          "R3X = cuticle / soft-tissue extension of R3 (in progress); INFRA = run-contract tests.\n")
md.append("| run_dir | date (UTC) | node | title | status | feeds | orx outcome | r2_verified | objects | MB | note |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    md.append("| `{}` | {} | `{}` | {} | **{}** | {} | {} | {} | {} | {} | {} |".format(
        r["run_dir"], r["date_utc"], r["orx_node"][:8] or "-", esc(r["node_title"]), r["status"], esc(r["feeds"]),
        r["orx_outcome"], r["r2_verified"], r["n_objects"], r["size_MB"], esc(r["note"])))

# appendix: orx runs without R2 folder, nodes without runs
have = {x["orx_run"] for x in merged}
md.append("\n## orx runs without an R2 folder\n")
md.append("| orx run | orx outcome | node title | note |\n|---|---|---|---|")
notes = {"bf8497cd": "finder v2 fold 2 on pod8 (no network); stale 'Cancelling' in orx",
         "576548ec": "finder v2 fold 2 on pod8: failed on a stale uv cache lock before any upload",
         "7cbcdbb4": "baseline smoke test (root node); reason unknown (earliest run, before node f80139b3 added run metadata)"}
for k, v in orx.items():
    if k not in have:
        md.append(f"| `{k[:8]}` | {v['status'].replace('…','').lower()} | {esc(v['title'])} | {notes.get(k[:8], 'unknown')} |")
md.append("\n## orx nodes that never ran (no R2 folder)\n")
md.append("- `4824569e` nnU-Net 2d fold0 on Toulkeridou labels (replaced by `ef2248f0` / `2ea0308a`).\n"
          "- Finder v3b folds 1, 3 (`69e1aaef`, `2719a9c6`); finder v3c folds 0-4 (`962d3101`, `ab7466cf`, `df6d17a0`, `ccd474a9`, `4451f56d`: dataset v5 cancelled); "
          "finder v3d folds 0-4 (`2e25d2e0`, `36561afc`, `05f0f736`, `fa77ee27`, `2da52506`: replaced by one fold-all finder).\n"
          "- Cuticle Matte v1 full shards 1/5-4/5 (`eb53ef09`, `2d7d1c4d`, `b9ca86e7`, `f396a97a`: full-stack downloads infeasible).\n"
          "- Root `791a9064` ran once (`7cbcdbb4`, see above).\n")
(OUT / "RUNS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(len(rows), dict(cnt), tot_n, round(tot_mb, 1))
