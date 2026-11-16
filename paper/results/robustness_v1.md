# ANT-TWIN — preregistered robustness checks R2, R8 and the measurement-error model, v1 (15 Nov 2026)

Completes the checks that were "not run" in `primary_v1.md`. Prereg: `prereg/prereg_draft.md` sec. 6–8, `prereg_v1.0_frozen.md`.
Inputs: primary run R2 `ant-twin/runs/20261115-100746_brain-analysis-v1-primary-unblinded-preregistere_DESKTOP-7Q7S0N3/outputs/`
(`primary_data.csv`, `species_table.csv`, `specimens_table.csv`). The primary model, SESOI (0.02), decision rules and the frozen brain table are unchanged.
Before any check the primary MCC fit was reproduced locally: β₂ = +0.0351 [0.0166; 0.0535], n = 270 (identical to `primary.json`).

## Summary

| Check | β₂ (log10 colony) | 95% interval | n | Outcome (sec. 7 rules) |
|---|---|---|---|---|
| Primary (reference) | +0.035 | [+0.017; +0.054] | 270 species | support H2 |
| **R8** no supercolonial / social parasites, MCC | **+0.040** | [+0.020; +0.061], p = 0.0001 | 251 species | **support H2** |
| R8, 100 posterior trees (pooled) | median +0.038 | [+0.018; +0.059], all p < 0.05 | 251 | support H2 |
| **R2** Borowiec 2025 tree, one species per genus, 100 resamples | **median +0.029** (2.5–97.5% of resamples +0.017…+0.040) | pooled [+0.001; +0.058] | 117 genera | **support H2** (77/100 resamples H2, 23 undecided, 0 negative) |
| **ME model** (brms, specimen level, m = 0.031) | **+0.038** (posterior mean) | 95% CrI [+0.019; +0.055]; 90% [+0.023; +0.053] | 442 specimens / 270 species | **support H2**; P(β₂ > 0) = 1.000, P(β₂ > 0.02) = 0.97 |

**Reading.** All three checks keep the primary outcome (support H2). No estimate in any check, tree or resample is negative.
The brms interval agrees with the primary result, so the label **"support H2, robust"** of sec. 7 (posterior trees + brms) now holds in full.
R2 is weaker only because it uses ~117 genera instead of 270 species (see the exploratory control below), not because of the tree.

## R8 — without supercolonial species and social parasites (preregistered; done)

- Flags: Bell-Roberts et al. 2024, Supplementary Table 21 ("Ant species exclusions"; 112 species; 33 supercolonial, 25 social parasites), from the authors' GitHub `Supplementary_Tables.xlsx` (the same file the specimen-table stage reads for Table 20). Names normalised with `specimen_table.key`.
- 19 of the 270 primary species are flagged and dropped: supercolonial — *Anoplolepis gracilipes, Cardiocondyla emeryi, C. obscurior, Formica polyctena, F. truncorum, Lasius neglectus, L. sakagamii, Linepithema humile, Monomorium pharaonis, Paratrechina longicornis, Pheidole megacephala, Solenopsis invicta, Tapinoma melanocephalum, Wasmannia auropunctata*; social parasites — *Formica sanguinea, Formicoxenus nitidulus, Polyergus rufescens, Strongylognathus testaceus*; both — *Myrmica rubra*.
- Species not listed in Table 21 are treated as unflagged (the table only covers Bell-Roberts' species set).
- Result: MCC β₂ = +0.0404 [0.0199; 0.0608], 90% CI [0.0232; 0.0575], λ = 0.74, n = 251 → support H2. 100 NC-stem posterior trees: median +0.0385, pooled 95% [0.0181; 0.0589], all p < 0.05 → support H2 (robust over trees).

## R2 — Borowiec et al. 2025 tree (preregistered; done)

- Source: Zenodo 14338504 (CC BY 4.0), `10-MCMCTree.zip`, dated MCMCTree chronograms (292 tips, 289 genera, incl. non-ant outgroups). Local copy `work/borowiec2025/`; R2 copy `ant-twin/meta/borowiec2025/` (zips, extracted FigTree trees, consensus tree, README, `manifest.json` with SHA-256).
- The tree is genus-level (one exemplar per genus; 2 genera have 2 tips). Prereg rule applied: per resample one random species per genus, attached to that genus' tip (random tip if two); 100 resamples (seed 2025); median and spread of β₂ reported; pooled interval = t-mixture over resamples (as for posterior trees). Species of genera split after the tree's taxonomy join the former genus via the frozen `pgls.GENUS_TRANSFERS` (*Holcoponera, Poneracantha, Stictoponera* → *Gnamptogenys*).
- VCV is taken from the MRCA of the sampled tips (outgroup stem removed, as `ape::keep.tip`).
- Species pool: all 272 species with brain + body + colony (the 270 primary species + *Gesomyrmex howardi* and *Gigantiops destructor*, which are absent from the Economo tree but present here). 117 of 125 genera matched; missing genera: *Buniapone, Euponera, Mystrium, Ophthalmopone, Streblognathus* (6 species).
- Main tree: root prior 129–158 Ma, independent rates (chosen before looking at results; the other three dating schemes are sensitivity):

| Tree / set | Median β₂ | 2.5–97.5% of resamples | Pooled 95% | Share H2 / undecided | λ median |
|---|---|---|---|---|---|
| 129–158 Ma independent, 272 spp, **100 resamples (prereg)** | **+0.029** | [0.017; 0.040] | **[+0.001; +0.058]** | 77% / 23% | 0.50 |
| same, 1000 resamples | +0.029 | [0.018; 0.043] | [+0.001; +0.059] | 72% / 28% | 0.45 |
| same, 270 primary species (115 genera) | +0.032 | [0.021; 0.043] | [+0.003; +0.061] | 81% / 19% | 0.35 |
| 129–158 Ma correlated | +0.030 | [0.018; 0.042] | [+0.002; +0.059] | 80% / 20% | 0.56 |
| <224 Ma independent | +0.029 | [0.017; 0.040] | [+0.001; +0.058] | 76% / 24% | 0.52 |
| <224 Ma correlated | +0.029 | [0.018; 0.042] | [+0.001; +0.059] | 79% / 21% | 0.55 |

- Outcome of the pooled interval: support H2 in every tree. Per resample the 95% CI never lies below 0; the median per-resample CI is [+0.003; +0.055].
- **Exploratory control (not preregistered):** the same 100 one-species-per-genus draws on the 270 primary species, but each species keeps its own tip in the Economo MCC tree: median +0.031 (2.5–97.5%: 0.018–0.043), pooled [+0.002; +0.061], 77% H2 / 23% undecided. This is the same as on the Borowiec tree, so the wider R2 interval comes from reducing n to ~115 genera, not from the topology or dating.

## Measurement-error model (preregistered sec. 6; done in brms as planned)

- Toolchain: brms 2.23.0 + CmdStan 2.37.0 (cmdstanr backend), built with Rtools45 installed in the user profile. The rstan backend failed to compile on R 4.6.2, so cmdstanr was used — same brms model, not a deviation in the model.
- Data: the 442 specimens averaged into the 270 primary species means (main_ok, body present, the species' chosen stain; the specimen-level log10 means reproduce `primary_data.csv` exactly). Specimen-level log10 body, stain and scan-step class; species-level log10 colony.
- Model exactly as preregistered: `log10_brain | se(m, sigma = TRUE) ~ log10_body + log10_colony + iodine + vox1.22 + vox6.11 + (1 | gr(phylo, cov = A)) + (1 | species)`, A = correlation matrix of the Economo MCC tree pruned to the 270 tips; priors b ~ N(0, 0.5), Intercept ~ t₃(0, 2.5), SDs and σ ~ half-t₃(0, 0.5); 4 chains, adapt_delta 0.95.
- **Known measurement SD m.** Seg v3 held-out relative brain-volume error, median |error| converted to an SD (÷ 0.6745) and to log10:
  - main: AntScan held-out (gate-1 drafts), median |error| ≈ 5% (folds 3.0–8.9%) → **m = log10(1 + 0.05/0.6745) = 0.0311** for every specimen (all specimens are AntScan scans);
  - sensitivity: Toulkeridou human labels, median ≈ 2% (folds 1.0–2.6%) → m = 0.0127.
- Convergence. With the preregistered 4 × 2000 iterations (1000 warm-up): β₂ = +0.0375 [0.0193; 0.0554], max R-hat 1.007, 0 divergences, but the minimum bulk ESS was 372 (the small non-phylogenetic species SD) < 400. The run was therefore extended to 4 × 4000 iterations (same warm-up, seed, priors): max R-hat 1.004, min bulk ESS 1091, min tail ESS 2335, 0 divergences — all criteria met. Reported values are from this run.

| m | β₂ mean [95% CrI] | 90% CrI | P(β₂ > 0.02) | β body | SD phylo | SD species | σ |
|---|---|---|---|---|---|---|---|
| **0.0311 (main)** | **+0.0377 [0.0193; 0.0555]** | [0.0226; 0.0527] | 0.972 | 0.451 | 0.174 | 0.030 | 0.139 |
| 0.0127 | +0.0376 [0.0197; 0.0561] | [0.0225; 0.0527] | 0.973 | 0.452 | 0.174 | 0.029 | 0.142 |

- The colony coefficient does not depend on the assumed measurement error. Most within-species variance is real worker variation (σ ≈ 0.14 log10), not segmentation error (m ≤ 0.031).
- MCMCglmm 2.36 cross-check (preregistered fallback, run in addition; same fixed and random effects, mev = m² with m = 0.0311, parameter-expanded priors, 4 chains × 260k iterations, burn-in 10k, thin 100): β₂ = +0.0378 [0.0197; 0.0559], P(β₂ > 0) = 1.000; ESS ≈ 10 000 for all fixed effects and variance components, PSRF ≤ 1.0013 (the fixed measurement-variance term has no ESS by design). Matches brms.

## Deviations (sec. 11 log; all after the colony join)

1. **ME model iterations:** 4 × 4000 instead of 4 × 2000, because one variance component had bulk ESS 372 < 400 with 2000. The 2000-iteration estimate (+0.0375 [0.0193; 0.0554]) is reported too; the conclusion is the same.
2. **ME model, m:** prereg says "held-out error plus between-fold spread". The seg v3 ensemble did not record per-specimen fold spread (`fold_cv` empty), so m is the held-out error only (main 0.0311, sensitivity 0.0127).
3. **ME model backend:** cmdstanr instead of rstan (rstan failed to compile on R 4.6.2). Model and priors unchanged.
4. **R2 species pool:** 272 species (adds 2 species placed in the Borowiec but not the Economo tree), since tree placement is part of the inclusion rule; the 270-species version is reported alongside. Main dating scheme fixed to 129–158 Ma / independent rates; the other three are shown.
5. **R8 flags:** only Bell-Roberts Table 21 as preregistered; species outside Bell-Roberts' set are counted as unflagged.

## Files

- Scripts: `research/ant-twin/scripts/robustness_v1/` — `common.py`, `r8_run.py`, `r2_run.py`, `r2_economo_genus.py` (exploratory), `me_prep.py`, `me_brms.R`, `me_mcmcglmm.R`.
- Reusable code (ant-twin worktree, not committed there): `src/analysis/robustness_extra.py` (Bell-Roberts flags, R8 subset, NEXUS reader, genus resampler, resample summary).
- Outputs (local, `work/robustness/`, git-ignored): `r8.json`, `r2.json`, `r2_reps_*.csv`, `r2_economo_genus.json`, `me_data.csv`, `me/brms_summary_*.csv`, `me/brms_log_*.txt`, `me/mcmcglmm_summary_m_main.csv`.
