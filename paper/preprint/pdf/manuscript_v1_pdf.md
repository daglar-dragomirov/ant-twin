# Workers of large-colony ants do not have smaller brains: a preregistered test

**Authors:** [AUTHOR LIST AND AFFILIATIONS — to be completed by the author]

**Correspondence:** [e-mail]

**Status:** draft v1, 16 November 2026. The first draft (manuscript.md, v0) rewritten with the paperforge pipeline: structure and wording changed, numbers and citations unchanged. All numbers come from the committed result files listed in the *Data and code availability statement*. Values marked "draft check" were computed for this draft with `preprint/tables/supp_checks.py`, which runs the frozen analysis code read-only. Abstract: 218 words.

**Keywords:** ants, brain size, colony size, social brain hypothesis, distributed cognition, micro-CT, deep learning, phylogenetic comparative methods, preregistration

## Abstract

Large insect societies make many decisions collectively, which could relax cognitive demands on individual workers and favour smaller brains (the distributed-cognition hypothesis); the social-brain hypothesis predicts the opposite. We tested these predictions in 270 ant species with a preregistered phylogenetic model, using worker brains segmented automatically from AntScan synchrotron micro-CT scans and checked by a vision-language-model agent. Relative brain volume increased with colony size. The increase was about 8% per tenfold increase in colony size, and the 95% confidence interval excluded zero, which met the preregistered criterion for the social-brain hypothesis. The result held across posterior trees, a second phylogeny, a model of measurement error and all robustness checks except the soft-tissue denominator. Exploratory analyses showed that the effect size depends on the body measure: with independently measured body volumes, or with soft tissue (body minus cuticle), it shrank to a small, uncertain value. Thinner cuticle in large-colony species explained only a small part of this difference; most of it traced to errors in a few of our own body masks. In every analysis in Table 1, estimates were positive and intervals excluded a decrease as large as the preregistered smallest effect of interest. Taken together, our results show that workers of large-colony species do not have smaller brains: brain investment is maintained or at most modestly increased.

## 1. Introduction

Colony size varies widely in ants: species range from colonies of a few workers to millions, and in our data set species means span 7 to 10⁷ workers. It evolves gradually across the ant phylogeny (Burchill & Moreau 2016), and group size shapes how colonies are organised, including division of labour and collective decision-making (Dornhaus et al. 2012). A precedent comes from cuticle: across ants, workers of species with larger colonies invest less in cuticle. Matte et al. (2025) described these as "cheaper workers" and argued that this facilitated larger societies. This finding raises a direct question: is the per-worker economy limited to protective tissue, or does it extend to neural tissue? Two classic ideas give opposite answers. Under the distributed-cognition hypothesis (H1), information processing in large colonies is spread across many workers, so each worker may need less individual processing (Feinerman & Traniello 2016; O'Donnell et al. 2015). Selection could then reduce its brain investment. The social-brain hypothesis (H2) predicts the opposite, because larger and more complex societies may demand more processing of social information from each member (Dunbar 1998; Gronenberg & Riveros 2009; Lihoreau et al. 2012).

The evidence so far comes from few species, mostly within one genus or tribe, and some studies examined specific neuropils rather than total brain volume (Section 1.1). To our knowledge, no study has tested colony size against worker brain size across hundreds of ant species within a phylogenetic framework.

We ask whether workers of species with larger colonies have smaller brains for their body size. Our answer is that they do not: across 270 species, the distributed-cognition prediction of smaller brains was not supported by any analysis. The preregistered test instead found a positive association, but its magnitude depends on how body volume is measured.

AntScan, a synchrotron micro-CT survey of ant specimens that spans 792 species (Katzke et al. 2026), allows brain volumes to be measured across species at a scale that manual segmentation cannot reach. On these scans we built an automated pipeline of brain localisation, nnU-Net segmentation (Isensee et al. 2021) and vision-agent quality control, and froze the resulting brain table. Figure 1 shows this pipeline on real scans, from the whole specimen to the brain mask, and Supplementary Fig. S1 gives a schematic of the measurement and analysis pipeline. We then preregistered one primary test of the colony-size slope, with a smallest effect size of interest (SESOI) and explicit decision rules, and ran the preregistered robustness checks. Exploratory analyses separated brain investment relative to soft tissue from investment relative to total body volume. Figure 3 shows the colony-size estimate in the primary analysis and in every robustness check.

To test this claim, we ask three questions.

- **Q1** *Are the automated brain volumes accurate enough for a comparative test?* Yes: segmentation matched held-out human labels closely, brain volume was repeatable within species, and both preregistered gates were passed (Section 3.1).
- **Q2** *Do workers of large-colony species have smaller brains for their body size?* No: in the preregistered test, relative brain volume increased with colony size, and all preregistered robustness checks except the soft-tissue denominator gave the same outcome (Section 3.2).
- **Q3** *Does the estimate depend on how body size is measured?* Yes: with the most accurate size measures, the estimate shrank to a small, uncertain value, and thinner cuticle explained only a small part of the difference (Section 3.3).

**Contributions.** This paper makes three contributions.

- **A comparative data set of worker brain volumes.** We measured brain volume automatically in workers of 325 species, with segmentation validated against human labels and every mask reviewed by a vision-language-model agent (Sections 2.2 and 3.1).
- **A preregistered phylogenetic test of the colony-size prediction.** Colony size was tested against relative brain volume with a smallest effect size of interest and decision rules fixed before unblinding (Sections 2.6 and 3.2).
- **A decomposition by body-size measure.** Measuring body and soft tissue with the same instrument on the same specimens separates the contribution of thinner cuticle from differences between body-volume sources (Section 3.3).

Section 1.1 relates the two hypotheses to earlier tests, and Section 2 describes the measurements and the analysis plan. The three questions are answered in Section 3, and Section 4 discusses what the results do and do not show.

### 1.1 Related work: two hypotheses and earlier tests

**Distributed cognition and the cost of neural tissue.** Neural tissue is metabolically expensive (Aiello & Wheeler 1995; Niven & Laughlin 2008), so selection could reduce brain investment when the colony takes over part of the computation. In vespid wasps, mushroom-body investment fell at the origins of sociality (O'Donnell et al. 2015). Among fungus-growing ants, relative brain size was smaller in monomorphic species with large colonies (Riveros et al. 2012). Unlike these studies within single clades, we test the prediction on whole-brain volume across 270 species from 12 subfamilies.

**Social brains in ants.** Some comparisons in ants point towards the social-brain prediction. Within *Cataglyphis*, species with larger colonies have larger brains (Wehner et al. 2007), and socially complex *Oecophylla* invests more in brain tissue than *Formica* (Kamhi et al. 2016). Rather than comparing species within one genus or two genera, we estimate the colony-size slope across the family in a phylogenetic model with a preregistered smallest effect of interest.

**Direct tests of colony size and brain size.** Such tests used few species, for example 2 (Kamhi et al. 2016), 3 (Godfrey & Gronenberg 2020) or 18 (Riveros et al. 2012), mostly within one genus or tribe. In harvester ants (*Pogonomyrmex*), species with larger colonies did not have larger brains (Perl et al. 2026). Other work examined specific neuropils, such as antennal lobes, rather than total brain volume (Godfrey & Gronenberg 2020; Marty et al. 2025). In contrast, we measure total brain volume in workers of hundreds of species from the AntScan survey.

**Cheaper workers.** Most closely related to our work is the comparative study of cuticle investment by Matte et al. (2025), who measured body and cuticle volumes on AntScan specimens and used the same phylogeny. Unlike Matte et al. (2025), whose measurements concern protective tissue, we measure investment in neural tissue and use their body and cuticle measurements as size denominators.

## 2. Methods

### 2.1 Study design and specimens

Our comparison combines four quantities: species means of worker brain volume and body volume, the colony size of each species, and its position on the ant phylogeny. We use the term *relative brain volume* for brain volume at equal body volume, stain and scan step. Accordingly, we write β₂ for the colony-size slope, the change in log10 brain volume per log10 unit of colony size at equal body volume, stain and scan step. By *soft tissue* we mean body volume minus cuticle volume.

Specimens come from AntScan, synchrotron micro-computed tomography (micro-CT) scans made at the Karlsruhe Institute of Technology (KIT). The scans are hosted on Biomedisa under a Creative Commons Attribution licence (CC BY 4.0; Katzke et al. 2026; Lösel et al. 2020). The target set comprised all colony workers (caste "worker" in the AntScan metadata) of species with a species-level colony size: 818 specimens of 352 species. Scans differ in voxel size (1.22, 2.44 or 6.11 µm; 8.2 µm was merged with 6.11 µm) and in staining (iodine or unstained). Both are partly confounded with body size, so both enter the model as covariates (Section 2.6). Appendix B describes how scan stacks were read.

### 2.2 Brain volume: localisation, segmentation and quality control

**Localisation and segmentation.** The brain pipeline has three steps: localisation, segmentation and quality control. First, a low-resolution "finder" (nnU-Net 3d, trained on 471 of 497 approved low-resolution brains, resampled to isotropic 12 µm and cropped to the specimen) located the brain in each specimen. A full-resolution box was then cut around the head (Fig. 1a); boxes that cut off part of the brain were regrown, and unresolved cases received second and third finder candidates. Brains were then segmented with seg v3, a 5-fold nnU-Net 2d ensemble trained on three sources:

- 117 micro-CT stacks with human brain labels from Toulkeridou et al. (2023);
- 24 AntScan brains from a bootstrap round ("gate-1"), whose reference masks are automatic drafts approved by vision agents;
- 530 AntScan brains approved by vision-agent quality control in earlier rounds.

Folds were assigned by species within each data source, and the final segmentation of every specimen is the 5-fold ensemble without test-time augmentation (Fig. 1b–c). Accuracy was assessed on the held-out folds against each kind of reference (Section 3.1). "Brain" follows the Toulkeridou et al. (2023) label: all brain tissue in the head capsule, including the subesophageal zone. The quality-control notes refer to the subesophageal ganglion as part of the mask **[CHECK: confirm exact label definition from the Toulkeridou masks; the paper does not state it]**.

**Quality control (QC).** No human reviewed the masks. Instead, every final mask was reviewed by a multimodal language-model agent (Claude Opus 5.5, Anthropic; high reasoning effort). It saw a contact sheet of eight slices through the head box, with the mask outline overlaid. The agent judged whether the box is on the head, whether the mask is good, needs fixing or is bad, and whether the brain lies fully inside the box, and it estimated the volume error. A specimen entered the brain table only with "mask good", "box on head" and "brain fully in box"; Supplementary Fig. S4 shows rejected masks with the agent's verdicts. Verdicts, prompts and images are archived (*Data and code availability statement*), and the table was frozen before unblinding (checksum in Appendix B).

### 2.3 Body volume and soft tissue

**Body volume.** This preregistered primary size covariate came from two sources, in this order. Where a specimen had been measured by Matte et al. (2025), we used their individual measurement (their Dataset S1; hereafter owner-measured volumes). Otherwise we used our own body mask (v8): a three-class multi-Otsu segmentation inside the vial, with per-slice contour closing and filling, and correct slice spacing from the true page counts. These masks agreed with the owner-measured volumes with a median ratio of 1.04 (n = 278 specimens), and specimens with 20 or more "guarded" slices (bright mounting medium) had no valid body volume. Of the 462 main-analysis workers with a body volume, 235 had owner-measured values and 227 had values from our masks.

**Soft tissue.** Robustness check R3 used owner-measured soft tissue, available for 154 of the 270 primary species. The preregistration anticipated adding our own cuticle measurements "if time permits", and we did so in an extension declared after unblinding. For this extension we ran the owners' own segmentation program (Matte et al. 2025; Zenodo record 15546461, CC BY 4.0) unchanged on 185 specimens. These were 141 target workers (at most two per species) from the 116 primary species without owner-measured soft tissue, and 44 owner-measured specimens for validation.

**Subsampled slices.** Because the AntScan server throttles downloads, only the slices actually used were read, by HTTP range requests: every third slice, plus every thirtieth slice for the program's z-limit search. The program's median filter therefore spans 9 sampled slices instead of 25. On 8 specimens processed at every slice, the ratio of every-third-slice to every-slice volumes had a median of 0.9996 for body (range 0.990–1.010). For soft tissue, the median was 0.9991 (range 0.978–1.002).

**Threshold rules.** With the authors' threshold (110), a bright mounting medium leaks into the mask in many scans, and the authors' scripts contain both 110 and 145. Our main ("adaptive") rule therefore used threshold 110 if body(110)/body(130) lay within 1/1.10–1.10 and otherwise 145 if body(145)/body(130) lay within that range. Specimens that met neither condition were excluded (Supplementary Fig. S3), and in all cases the rule required cuticle/body between 0.03 and 0.40. A "strict" rule, used as a sensitivity analysis, allowed threshold 110 only, with body volume stable across all three thresholds (max/min ≤ 1.15). The adaptive rule passed 141 of 185 specimens (threshold 110: 92; threshold 145: 49), and the strict rule passed 88.

**Validation.** Of the 44 validation specimens, the adaptive rule accepted 41. On these, our measurements reproduce the owners' values, with median ratios of 0.998 for body, 1.04 for cuticle and 0.992 for soft tissue. The correlation of log soft tissue was r = 0.997, and the three excluded validation specimens are the ones that disagree with the owner-measured values.

### 2.4 Colony size and phylogeny

**Colony size.** Species colony size is the geometric mean of the mean nest sizes recorded in Dataset S3 of Matte et al. (2025). For species without S3 records, we used the colony size from Bell-Roberts et al. (2024); this applied to 6 of the 270 primary species. Across the 270 primary species, colony size ranges from 7 to 10⁷ workers (median 300; SD of log10 colony size 1.24).

**Phylogeny.** The primary tree is the maximum-clade-credibility (MCC) tree of the Economo et al. (2018) ~15,000-species phylogeny (NC-uniform, stem); Matte et al. (2025) used the same phylogeny (mean covariance of its FBD trees). Tree uncertainty was assessed on 100 NC-stem posterior trees (preregistered). Alternative trees were 200 FBD crown and stem posterior trees (check R1) and the genus-level dated phylogeny of Borowiec et al. (2025) (check R2). Species were placed on the tree by exact name, then subspecies, then a fixed list of post-2018 genus transfers, and then a free tip of the same genus. Of the 270 primary species, 257 matched by exact name, 12 via genus transfer and 1 by genus placement.

### 2.5 What was fixed before unblinding, and what was not

The analysis plan and outcome rules were preregistered (OSF "secondary data analysis" template) as `prereg_draft.md` v0.1 on 13 November 2026. The plan was frozen, with declared deviations, on 15 November 2026, before any model containing colony size was fitted to real brain volumes (commit and tag in Appendix B). Measurement stages never read the colony table, and until the freeze, analysis runs wrote only a "has colony" flag next to brain values. The deviations declared at the freeze concern the segmentation model, mask review by vision agents instead of humans, and body volumes from v8 instead of v6 masks. They also concern registration as a timestamped GitHub commit and tag instead of OSF, and running the measurement-error model after the primary test. After unblinding, the measurement protocol and QC rule of the soft-tissue extension were specified; the rule was fixed before the final fits and revised once, on segmentation diagnostics. An interim estimate on 28 added species (+0.007) had been seen before that rule was written, so both rule versions are reported. Appendix A lists every deviation. Finally, the test concerns the whole-brain volume of minor and monomorphic workers; it does not address brain composition, such as the mushroom bodies.

### 2.6 Statistical model and decision rules

**Worker selection.** Species values are means of log10 volumes over workers of one preparation that passed quality control: unstained workers if any, otherwise iodine-stained ones, and the modal scan-step class. Only minor and monomorphic workers were used, so major, media and intercaste workers were excluded, as were workers with damage that affects the brain (head or anterior truncation, oversaturation, badly preserved soft tissue).

**Primary model.** The primary model is a phylogenetic generalised least squares (PGLS) regression of log10 brain volume on log10 body volume, log10 colony size, stain and scan step:

log10 Brain = β₀ + β₁·log10 Body + β₂·log10 Colony + β₃·Iodine + β₄·Step1.22 + β₅·Step6.11 + ε, with ε ~ N(0, σ²C_λ).

In words, β₁ is the brain–body slope, β₂ the colony-size slope, and β₃ to β₅ adjust for staining and scan step. Here C_λ is the Pagel-λ transformed MCC covariance (Pagel 1999; Freckleton et al. 2002), with λ estimated by maximum likelihood; 95% CIs for β₂ are t-intervals with n − 6 degrees of freedom. We fitted the model in Python with `pgls.pgls_lambda` and cross-checked it against `phylolm` (Ho & Ané 2014) and `nlme::gls(corPagel)`; differences were below 10⁻⁶. For tree uncertainty, the model was refitted on the 100 posterior trees and summarised by the median and a pooled interval (a mixture of the per-tree t sampling distributions).

**Gates before unblinding.** Before unblinding, the data had to pass two gates. The Haller's-rule positive control, a PGLS-λ fit of log10 brain on log10 body, stain and scan step, required a slope in [0.4, 0.9], a CI inside (0, 1) and R² > 0.5. In the synthetic-recovery gate, brains were simulated on the real tree, species set and colony sizes with β₂ = −0.03 (200 replicates).

**Sample size and precision.** The sample was fixed by data availability: every AntScan species with colony size, an approved brain, body volume and a tip in the tree (expected 310–340 species in the preregistration; 270 realised). The preregistration justified this size by simulation on the real MCC tree and colony sizes (λ = 0.8, brain–body slope 0.7, measurement noise 0.03 log10, 1000 replicates; Supplementary Table S10). With a species-level residual SD of 0.10–0.15 the expected SE(β₂) was 0.004–0.006, giving a power of at least 0.91 to detect β₂ = −0.02 and of at least 0.93 to declare equivalence within ±0.02 when β₂ = 0; with a residual SD of 0.25 (SE 0.010) these powers fall to 0.53 and 0.33. The synthetic-recovery gate on the realised species set confirmed unbiased recovery (see Results). In the real data the SE of β₂ was 0.0094 (n = 270), so the half-width of the 95% CI (about 0.018) is smaller than the SESOI: the design resolves effects of the size of the SESOI, and an interval that excludes −0.02 reflects the data rather than low precision.

**Decision rules.** The smallest effect size of interest was |β₂| = 0.02, about 4.7% brain volume per tenfold difference in colony size. Outcomes were labelled as follows.

| Outcome | Condition |
|---|---|
| Support H1 | 95% CI entirely below 0 |
| Support H2 | 95% CI entirely above 0 |
| Equivalence (H0) | 90% CI inside (−0.02, 0.02); a two one-sided test (Lakens 2017) |
| Small but non-zero | CI excludes 0 and lies inside ±0.02 |
| Undecided | otherwise |

The label "robust" is added if the posterior-tree interval gives the same outcome and the measurement-error model does not contradict it.

### 2.7 Robustness checks and exploratory analyses

**Preregistered robustness checks.** The 14 preregistered robustness checks R1–R14 vary the phylogeny, the size measure, the species and worker sets, the colony-size measure and the phylogenetic model (Table 1):

| Check | Variant |
|---|---|
| R1 | FBD posterior trees |
| R2 | Borowiec et al. (2025) tree, one random species per genus, 100 resamples |
| R3 | Soft tissue instead of body volume |
| R4 | Leaving out each subfamily with ≥ 10 species |
| R5 | Unstained species only |
| R6 | All workers, including majors |
| R7 | Without names qualified by cf./aff./nr. |
| R8 | Without supercolonial species and social parasites (Bell-Roberts et al. 2024, Supplementary Table 21) |
| R9 | Colony size from Bell-Roberts et al. (2024) only, or the maximum instead of the mean |
| R10 | Genus-level colony size |
| R11 | Without genus-placed tips |
| R12 | λ = 1, and the owners' method: mean covariance of FBD trees divided by 200 as a fixed correlation |
| R13 | Badly preserved specimens returned, with collection year as a covariate |
| R14 | OLS without phylogeny |

Appendix B gives the implementation details of R2 and R8.

**Measurement-error model.** In addition, a specimen-level measurement-error model (brms 2.23 with CmdStan 2.37; Bürkner 2017; Carpenter et al. 2017) used 442 specimens of the 270 species. Its phylogenetic covariance was the MCC correlation matrix A. The known measurement SD, m = log10(1 + 0.05/0.6745) = 0.0311, comes from the held-out AntScan volume error; m = 0.0127, from the human-labelled folds, served as a sensitivity analysis. An MCMCglmm fit (Hadfield 2010; 4 × 260,000 iterations) served as a cross-check. Appendix B gives the model formula, priors and sampling settings.

**Exploratory analyses.** Analyses not in the plan are reported as exploratory. The soft-tissue extension combined owner-measured soft tissue for 154 species with our measurements for 81, giving 235 species, and was fitted with the primary model on the MCC tree and on the 100 posterior trees. For the like-for-like decomposition (draft check), body and cuticle volumes were taken from the same instrument and the same specimens: owner-measured where available, otherwise the owners' program under the adaptive rule. The model was then fitted on the same species with this body volume, with soft tissue and with the primary-table body volume, which separates the cuticle contribution from differences between body-volume sources. These fits were repeated with λ fixed at 0 and at 1 for both denominators. The body-volume refits (draft check) rebuilt species means without workers whose brain/body volume ratio exceeds 10%, without workers above 5% and without all automatically flagged workers. Two further refits used owner-measured bodies only and our own bodies only. Their reference is the preregistered plausibility flag, which used the range 0.05–10%; it was recorded, but by the preregistered rules it triggers mask review rather than exclusion. R1 and R12 were also recomputed on all 200 FBD trees (draft check; Appendix A, item 11).

**Study size and question map.** Values marked "draft check" were computed for this draft with `preprint/tables/supp_checks.py`. The soft-tissue data set is smaller than the primary data set: owner-measured soft tissue exists for 154 of the 270 species, and our own measurements added 81 more. The measured workers of the other 35 species all failed QC. Q1 rests on the segmentation validation (Section 2.2) and the two gates (Section 2.6). We answer Q2 with the preregistered test and robustness checks and Q3 with the exploratory analyses.

## 3. Results

### 3.1 Q1: Are the automated brain volumes accurate enough for a comparative test?

The automated brain volumes are accurate enough for this test: segmentation matched held-out human labels closely, volumes were repeatable within species, and both preregistered gates were passed. On held-out folds, seg v3 agreed closely with human labels (Fig. 1d–e; Table S2; examples in Supplementary Fig. S2):

| Reference | Dice (median per fold) | \|volume error\| (median per fold) |
|---|---|---|
| Human labels (Toulkeridou et al. 2023) | 0.962–0.973 | 1.0–2.6% |
| AntScan gate-1 drafts | 0.917–0.956 | 3.0–8.9% (small n per fold) |
| Agent-approved AntScan masks | 0.982–0.985 | 1.0–1.2% |

The agent-approved references came from an earlier model, so the last row is not an independent accuracy estimate. For the 530 previously approved brains, the final ensemble masks agree closely with the approved masks. Their Dice median was 0.988, with a 10th percentile of 0.972, and their volume ratio median 0.996 (IQR 0.990–1.001). The bootstrap model trained only on Toulkeridou labels was also tested on 17 species absent from its training data (Dryad testing set): Dice median 0.965, |volume error| median 1.3%.

The frozen table holds 689 workers of 325 species (92% of the 352 target species). No brain was found for 129 specimens of 27 species, mostly scans with almost no soft-tissue contrast, damaged or empty specimens, and very small ants. Among the 175 species with at least two workers, the intraclass correlation of log10 brain volume, ICC(1), was 0.90. The median within-species coefficient of variation was 13% (IQR 6.7–26%), a figure that includes real worker polymorphism. In the Haller's-rule gate, brain volume scaled with body volume with a PGLS slope of 0.489 (95% CI 0.447–0.530; Fig. 2a). The ordinary least squares (OLS) R² was 0.85, with n = 270. This slope is within the preregistered range and consistent with the negative brain allometry known in ants (Seid et al. 2011; Wehner et al. 2007). The synthetic-recovery gate gave a bias of +0.0008, a root-mean-square error (RMSE) of 0.0037, a 95% coverage of 0.955 and a mean SE of 0.0038.

Of the 689 workers in the frozen table, 12 were excluded for damage and 49 as major, media or intercaste workers, which left 628 main-analysis workers, 462 of them with a body volume. The primary data set has 270 species from 123 genera and 12 subfamilies, 232 of them unstained (Table S1). Its species means rest on 442 workers, and 154 species are represented by a single worker. Brain volume ranges from 0.0024 to 0.50 mm³ and body volume from 0.0058 to 149 mm³.

### 3.2 Q2: Do workers of large-colony species have smaller brains for their body size?

No: in the preregistered test, relative brain volume increased with colony size. On the MCC tree, β₂ = +0.035 (95% CI 0.017–0.054; Fig. 2b), with a 90% CI of 0.020–0.051, p = 0.0002, λ = 0.72 and n = 270 species. This corresponds to about 8.4% more brain volume per tenfold increase in colony size at equal body volume, stain and scan step, and the 95% CI lies entirely above zero. Over the 100 posterior trees, the median β₂ was +0.033 (pooled 95% interval 0.014–0.051); per-tree estimates ranged from 0.030 to 0.035, and every tree gave p < 0.05. The measurement-error model gave a posterior mean of +0.038 (95% credible interval, CrI, 0.019–0.055), with P(β₂ > 0) = 1.000 and P(β₂ > 0.02) = 0.97. This estimate did not depend on the assumed measurement error: with m = 0.0127 it was +0.038 (0.020–0.056). Most within-species variance was real worker variation (σ = 0.139 log10 units) rather than segmentation error (m ≤ 0.031). An MCMCglmm cross-check gave +0.038 (0.020–0.056). The preregistered outcome is therefore support for H2 (robust); H1, smaller brains in larger colonies, is not supported.

All preregistered checks except R3 gave the same outcome, support for H2, with β₂ from +0.024 to +0.042 (Fig. 3; Table 1):

| Check | β₂ (95% interval) | Note |
|---|---|---|
| R1, FBD posterior trees | +0.032 (0.014–0.051) | on both 6 and 200 trees |
| R2, Borowiec tree | median +0.029 (pooled 0.001–0.058) | 117 genera; 2.5–97.5% of resamples 0.017–0.040 |
| R4, leaving out one subfamily | +0.024 to +0.042 | without Formicinae / without Myrmicinae |
| R8, without supercolonial species and social parasites | +0.040 (0.020–0.061) | n = 251 |
| R9, R10, alternative colony-size measures | +0.027 to +0.041 | |
| R12, λ = 1 | +0.040 | |
| R14, OLS without phylogeny | +0.027 (0.009–0.044) | |
| R3, owner-measured soft tissue | +0.006 (−0.013 to 0.024) | p = 0.56; n = 154; undecided |

Of the 100 R2 resamples, 77 supported H2 and 23 were undecided, and none gave a negative estimate. The wider interval comes from reducing n to ~117 genera rather than from the tree: the same one-species-per-genus draws on the Economo tree gave +0.031 (pooled 0.002–0.061; Table S4). The exception was R3: with owner-measured soft tissue as the size measure, the outcome was undecided.

### 3.3 Q3: Does the estimate depend on how body size is measured?

Yes: in the exploratory analyses, the estimate stayed near zero with soft tissue, and the gap to the body-volume estimate came mostly from the body-volume source rather than from thinner cuticle. Our measurements, made with the owners' program, raised soft-tissue coverage from 154 to 235 of the 270 primary species. With soft tissue as the size measure, β₂ stayed near zero under both QC rules (Fig. 4a; Table S6):

| Rule (species) | β₂ | 95% CI | 90% CI | Outcome |
|---|---|---|---|---|
| Adaptive (235) | +0.010 | −0.006 to 0.026 | −0.004 to 0.023 | undecided |
| Adaptive, 100 posterior trees | median +0.010 | −0.006 to 0.026 | n/a | no tree p < 0.05 |
| Strict (197) | +0.002 | −0.015 to 0.019 | −0.012 to 0.016 | equivalence (H0) |

In all soft-tissue fits, λ was estimated at its lower bound (≈ 0). On the same species, the primary-table body volume gave +0.034 (0.014–0.055; adaptive set) and +0.027 (0.007–0.047; strict set).

**Like-for-like decomposition.** The gap between body and soft-tissue denominators could reflect thinner cuticle in large-colony species (Matte et al. 2025). It could also reflect differences between the sources of body volume: our v8 masks versus owner-measured volumes or the owners' program. The like-for-like decomposition separates the two (Fig. 4a; Table S6). When body and soft tissue come from the same instrument and the same specimens, the two denominators give similar estimates:

| Species set | Body | Soft tissue |
|---|---|---|
| 235 species | +0.014 (−0.004 to 0.031) | +0.010 |
| 154 species with owner-measured soft tissue | +0.012 (−0.007 to 0.030) | +0.006 |

The cuticle contribution is therefore about +0.004 to +0.006, and the preregistration had anticipated a bias of about +0.005 from using body rather than soft tissue. The like-for-like body estimates are 0.008–0.020 below the primary-table body estimates on the same species (+0.020 and +0.034). This result did not depend on λ: with λ fixed at 0 or at 1 for both denominators, the like-for-like body and soft-tissue estimates differed by at most 0.006 (Table S6). For the 235 species, these estimates were +0.014 versus +0.010 at λ = 0 and +0.016 versus +0.015 at λ = 1. The gap between primary-table body and soft tissue also persisted at matched λ (Table S6).

**Thinner cuticle in large colonies.** Cuticle volume as a fraction of body volume declined with colony size (Fig. 4b), which reproduces the pattern of Matte et al. (2025) in our species set. The correlation r(log10 colony, log10 soft/body) was +0.26 for the 235 species and +0.39 for the 154 species with owner-measured soft tissue.

**Body-volume quality.** Brain/body ratios point to errors in a small number of v8 body volumes (Fig. 4c; Table S9). With owner-measured bodies, the median brain/body ratio was 1.95%, and no worker was above 10%. For v8 bodies, the median was 2.64%, and 6 workers were above 10%, with a maximum of 102%, which is physically impossible; 28 workers were above 5%. Refits of the primary model (Table 1, "draft check" rows) gave:

| Change to the primary data | β₂ | 95% CI | Species |
|---|---|---|---|
| Remove the 6 workers with brain/body > 10% | +0.024 | 0.007–0.042 | 268 |
| Remove workers with brain/body > 5% | +0.020 | 0.004–0.037 | 257 |
| Owner-measured (S1) bodies only | +0.010 | −0.011 to 0.031 | 155 |
| Own v8 bodies only | +0.035 | 0.010–0.060 | 162 |

The first two refits still support H2, whereas the fit with owner-measured bodies only is undecided. The brain–body slope was steeper with owner-measured bodies (0.645) than in the primary fit (0.473), which is consistent with measurement error in v8 bodies attenuating the size slope. Because colony size correlates weakly with body size (r = 0.13), part of the unexplained size signal can then load onto β₂. The six workers with brain/body > 10% belong to five species, among them two large-colony species (*Linepithema humile*, *Pheidole bicarinata*; Table S9).

**Across all analyses.** Every estimate in Table 1 was positive (minimum +0.002), and every lower 95% bound was above −0.02 (lowest −0.015). Negative effects of the size the preregistration deemed relevant are therefore excluded throughout. Only in the λ = 1 sensitivity fits of the 81 own-measured species (Table S6) were estimates slightly negative (−0.003), with wide intervals.

## 4. Discussion

**Main result.** Across 270 ant species, workers of species with larger colonies do not have smaller brains for their size. The distributed-cognition prediction of brain downsizing (H1) is not supported by any analysis. This holds for the preregistered test, 14 robustness checks, two phylogenies, a model of measurement error, two soft-tissue denominators and several body-quality filters. Every interval in Table 1 excludes a reduction as large as the preregistered SESOI (about 4.5% brain volume per tenfold colony size). For a comparative test across 123 genera, this is strong evidence against a general economy on neural tissue in large societies.

**A modest positive association.** The positive association (H2) is real by the preregistered criterion, but modest and sensitive to the body measure. The primary estimate (+0.035, about 8% per tenfold colony size) passed every preregistered robustness check except the soft-tissue denominator. Exploratory analyses that use the most accurate size measures put the effect at +0.006 to +0.014, with intervals that include zero. These measures are the owners' body volumes, body and soft tissue from the same instrument, and soft tissue. Part of the primary estimate is attributable to errors in a minority of our own body masks. We therefore read the data as showing that brain investment is maintained in large-colony species, possibly with a small increase, and we do not read them as strong support for the social-brain hypothesis. Remeasuring all body volumes with the owners' program (Section 4.2) would resolve how large the positive effect is.

**Cheaper workers, but not cheaper brains.** Matte et al. (2025) showed that workers of large-colony species invest less in cuticle, and our data reproduce that pattern. We find that this economy does not extend to the brain: relative to soft tissue, brain volume does not decline with colony size, and possibly increases slightly. Thinner cuticle accounts for only about 0.005 of the body-denominator slope, close to the preregistered expectation. Large societies thus economise on protective tissue but not on neural tissue.

**Relation to earlier studies.** Our broad result is consistent with Perl et al. (2026), who found no larger brains in *Pogonomyrmex* species with larger colonies. It also agrees with studies finding no colony-size effect on specific neuropils (Godfrey & Gronenberg 2020; Marty et al. 2025). In contrast, Riveros et al. (2012) found smaller relative brains in large-colony monomorphic attines (18 species). Narrow clades, specific ecologies or worker polymorphism may produce local patterns that do not generalise across the family. Distributed cognition may still act on brain composition rather than total volume, for example on mushroom bodies (O'Donnell et al. 2015; Kamhi et al. 2016, 2019). Whole-brain volume cannot test that, and neuropil-level segmentation of AntScan is a natural next step.

### 4.1 Limitations

Our study has several limitations.

- **AntScan scan classes.** Voxel size (1.22–6.11 µm) and staining differ between specimens and are partly confounded with body size. Both enter the model; the coefficient for the 1.22 µm class was −0.138 (−0.244 to −0.031; Table S3), and analyses restricted to unstained species (R5) gave the same outcome.
- **Colony-size data.** Colony sizes are species means compiled from the literature; they are often based on few records and vary within species. Error in the predictor tends to attenuate β₂ towards zero, which limits precision but is unlikely to create a positive estimate. Alternative colony measures (R9, R10) gave the same outcome.
- **Vision-agent QC instead of human review.** All inclusion decisions were made by a language-model agent; in the final round, one agent verdict per mask. The agent-approved masks agree closely with the final ensemble, and accuracy against human labels is high. Still, no human audit of a random sample has been done. **[AUTHOR: consider a blinded human audit of ~100 masks before submission]**
- **Body volume.** Our own body masks contain a few gross errors (Section 3.3). The preregistered pipeline flagged brain/body ratios outside 0.05–10% but did not exclude them, and this is the main reason why the effect size is uncertain.
- **Soft-tissue coverage.** Soft tissue is available for 235 of 270 species (197 under strict QC). The 35 missing species are those whose measured workers all failed QC, mostly because a bright mounting medium leaked into the mask (6.11 µm scans failed in 17 of 31 cases).
- **Cuticle QC rule.** The adaptive rule was written after an interim look (+0.007 on 28 species) and revised once, without looking at estimates under it. Both versions are reported, and both give estimates near zero (+0.010 and +0.002).
- **Subsampled slices.** Our runs of the owners' program read every third slice; the effect on volumes was below 0.1% (median), with a maximum deviation of 2.2% for soft tissue.
- **Departures from the preregistered plan.** R1 originally used 6 instead of 200 trees (recomputed here), R10 is an approximation, and the head-volume variant of R3 and the secondary analyses were not run.

### 4.2 Recommended next steps before submission (for the author; delete in final version)

1. Remeasure body and cuticle for all 270 primary species with the owners' program. This removes the v8 body masks from the primary analysis, as a clearly labelled exploratory analysis.
2. Run a blinded human audit of a random sample of masks.
3. Fix the brain-label definition.

### 4.3 Conclusion

We conclude that, across 270 ant species, workers of species with larger colonies do not have smaller brains. The preregistered test found a positive association between colony size and relative brain volume, but its magnitude shrinks to a small, uncertain value when the most accurate size measures are used. Negative effects of practical size are excluded in every analysis in Table 1. Large ant societies economise on worker cuticle but not on worker brains.

## Data and code availability statement

**Current state of the data.**

- **Results, intermediate outputs and model weights** are stored in a Cloudflare R2 bucket (`dsc-biology/ant-twin/`). This includes the frozen brain table, all masks, vision-agent verdicts and QC images, run logs, the primary-run outputs (`runs/20261115-100746_*`) and the robustness outputs (`analysis/robustness_v1/`, `analysis/r3_extension_v1/`).
- **Index.** An index of all stored objects is in `research/ant-twin/r2_index/INDEX.md` and `RUNS.md`.
- **Code** is in two GitHub repositories, currently private:
  - pipeline: `ant-twin`, one branch per run node;
  - write-ups, preregistration and robustness scripts: `dsc-biology`, `research/ant-twin/`.
- **Preregistration.** Frozen as tag `ant-twin-prereg-v1.0` (commit `c01e2ee`; freeze commit `3b1e350`).

**Plan for release.** Before posting:
1. make both repositories public;
2. archive a release on Zenodo with a DOI, containing the brain table, species tables, masks, QC verdicts and analysis code;
3. deposit a copy of the preregistration on OSF with the original commit date. **[AUTHOR: confirm]**

**Third-party data.**

- AntScan scans: CC BY 4.0 (Katzke et al. 2026).
- Matte et al. (2025) data and segmentation program: Zenodo 15546461, CC BY 4.0.
- Economo et al. (2018) trees: Dryad doi:10.5061/dryad.g579t7k, CC0.
- Borowiec et al. (2025) trees: Zenodo 14338504, CC BY 4.0.
- Toulkeridou et al. (2023) labels: Dryad doi:10.5061/dryad.qz612jmgv, CC0.
- Bell-Roberts et al. (2024): supplementary tables.

**Figures and tables of this draft** are generated by `preprint/figures/make_figures.py`, `preprint/figures/make_image_figures.py` (scan images: Fig. 1 and Supplementary Figs S2–S4), `preprint/tables/make_tables.py` and `preprint/tables/supp_checks.py`. Every image panel is rendered from data stored in the bucket; the selection rules, seeds and the bucket path of every input are recorded in `preprint/figures/image_data/image_figures_selection.json`, and the rendered figures are stored under `ant-twin/preprint/v1/figures/`.

## AI use statement

Large-language-model assistants (Claude, Anthropic) were used in this project as research tools, in the same way as other research software, with every step directed by the author. They helped design and write the measurement and analysis code (brain localisation, segmentation training and inference, the body- and cuticle-measurement wrappers, the PGLS and robustness scripts) under the author's direction; ran and monitored the cloud computations; drafted the preregistration and the analysis plan; summarised and interpreted results; and drafted, edited and translated text, which the author reviewed. They were also used for literature search, figure preparation and reference formatting.

In one place AI is part of the measurement protocol rather than an assistant: mask quality control was performed by vision-capable agents (Claude Opus 5.5) following a fixed written protocol (verdict "good" means that the box is on the head and the brain lies entirely inside it), with a second review of disputed masks. All verdicts and QC images are released (Section 2.2; Appendix A, deviation 2).

No claim of the paper rests on the assistants alone. Verification relied on checks that do not depend on the assistants' judgement: segmentation accuracy was measured against human expert labels of held-out species (Toulkeridou et al. 2023; Dice 0.962–0.973); body and soft-tissue volumes were validated against the owners' published measurements of the same specimens (Matte et al. 2025); the statistical model was cross-checked against independent R implementations (`nlme`, `phylolm`; differences below 10⁻⁶) and by synthetic recovery on the real tree; the analysis plan was frozen in a timestamped commit before unblinding; every reported number is regenerated from frozen records by the released scripts; and every reference was checked against Crossref or DataCite. The assistants supplied no measurements: all volumes are computed by deterministic code from the AntScan scans, and colony sizes and phylogenies come from published sources. The author takes full responsibility for the content of this paper, including any text, claims or code produced with the aid of these tools.

## Acknowledgements

[TO ADD.]

## Author contributions statement

[TO ADD.]

## Competing interests statement

[TO ADD.]

## Figure legends

![](figures/fig1_scan_to_brain.png){width=96%}

**Figure 1. From scan to brain volume.** Panels a–c show the automated pipeline on real AntScan scans; d–e show its held-out accuracy.

- **(a)** One specimen through brain localisation (*Labidus praedator*, Dorylinae, specimen 5x__22-04). Left: maximum-intensity projections (MIP) of the AntScan low-resolution copy that the finder reads (here 4.9 µm between slices, 9.4 µm in plane), with the full-resolution head box (blue) at the coordinates stored with the box; for this specimen the box had been regrown (Section 2.2). The ant hangs head-down in the vial. Right: middle slice of the head box at full resolution (2.44 µm voxels).
- **(b)** The brain mask of the same specimen (seg v3 ensemble, as frozen) as a surface inside its head box. The three orthogonal box slices through the mask centroid are drawn on the back walls with the mask outline. Orthographic view; for display only, the surface was computed on the mask downsampled twofold and smoothed. The scale bar lies along the front edge of the box floor.
- **(c)** Brain masks of eight specimens from the frozen brain table, shown without edits. Each tile shows the scan slice with the largest brain cross-section in the full-resolution head box, with the mask outline (orange) and a scale bar; the crop is centred on the mask, and white areas lie outside the stored box. Titles give species, subfamily and body volume (S1 or v8); the specimen key and voxel size are printed in each tile. *Selection rule:* the pool was the 462 main-analysis workers with a body volume, split into octiles of log10 body volume. From the smallest octile up, one specimen was drawn at random (NumPy `default_rng`, seed 20261116) among pool workers of subfamilies not yet drawn, so the eight specimens come from eight subfamilies. All eight happen to be 2.44 µm scans, the voxel class of 398 of the 462 pool workers. The specimen of panels a–b, framed in blue, is the drawn specimen closest to the pool median body volume.
- **(d)** Dice and **(e)** |relative volume error| of seg v3 on held-out folds (species-level folds within each source), by reference type:
  - human labels of Toulkeridou et al. (2023), n = 117;
  - AntScan gate-1 drafts, n = 24;
  - agent-approved AntScan masks, n = 530 (produced by an earlier model, hence not independent).

  Boxes show quartiles; numbers are medians. Axes are clipped: 8 cases below Dice 0.80 and 15 above 25% error are not shown.

*Data sources (bucket `dsc-biology/ant-twin/`):* low-resolution stack `cache/lowres/5x__22-04.npz`; head boxes `cache/boxes_v4/`, `boxes_grow/` and `boxes_v5/`; masks `cache/brain_masks_final/`, the ones counted in the frozen table (`runs/20261115-093934_*/outputs/brains.jsonl`; each mask's voxel count was checked against the frozen volume); held-out metrics `runs/20261114-212341_*` and `runs/20261114-215207_*` (`outputs/fold_k/val_per_case.json`). The path of every panel input is listed in `image_figures_selection.json` (folder `preprint/figures/image_data/`).

![](figures/fig2_allometry_colony.png){width=100%}

**Figure 2. Brain–body allometry and the colony-size effect.**

- **(a)** Species-mean log10 brain volume against log10 body volume for 270 species.
  - Brain volume is adjusted for stain and scan step with the coefficients of the Haller's-rule gate model.
  - Colour shows log10 colony size.
  - The line has the PGLS slope (0.489) and is drawn through the species means.
- **(b)** Partial residuals (component + residual) of log10 brain volume against log10 colony size from the preregistered PGLS fit on the MCC tree. The line shows β₂ = +0.035; the band is its 95% confidence interval.

![](figures/fig3_forest_beta2.png){width=100%}

**Figure 3. β₂ across the primary analysis and all robustness checks.**

- Points: estimates (for tree sets and resamples, medians).
- Lines: 95% intervals (pooled intervals for tree sets and resamples; credible intervals for the measurement-error models).
- Grey band: preregistered SESOI (±0.02). Dotted line: primary estimate.
- Colour gives the preregistered outcome label. Filled circles are preregistered analyses; open diamonds are exploratory analyses and draft checks.
- n is the number of species, or specimens/species for the measurement-error models.

![](figures/fig4_denominators_cuticle.png){width=100%}

**Figure 4. Size denominators, cuticle and body-volume quality.**

- **(a)** β₂ for the same species with three size measures:
  - body volume from the primary table (S1 or own v8 mask);
  - body volume from the same instrument and specimens as the soft-tissue measure (S1 or the owners' program);
  - soft tissue.

  Shown for the 154 species with S1 soft tissue and for the 235 species of the extended set. Lines are 95% CIs; the grey band is the SESOI.

- **(b)** Cuticle volume as a percentage of body volume (same specimens) against log10 colony size. Colour gives the source: S1 or our run of the owners' program. The line is OLS and descriptive only.
- **(c)** Brain/body volume ratio of main-analysis workers by body-volume source (log scale). The dashed line marks the preregistered plausibility limit of 10%.

In all legends, S1 denotes owner-measured volumes (Dataset S1 of Matte et al. 2025) and v8 our own body masks (Section 2.3).

## Supplementary figures

Supplementary Figs S2–S4 show cases chosen by fixed rules with a fixed seed, not the best-looking ones. The rules are stated in each legend and recorded, with all input paths, in `image_figures_selection.json` (folder `preprint/figures/image_data/`). Paths refer to the bucket `dsc-biology/ant-twin/`.

![](figures/figS1_pipeline_schematic.png){width=100%}

**Figure S1. Measurement and analysis pipeline (schematic).** AntScan synchrotron scans of colony workers pass through low-resolution brain localisation, a full-resolution head box, the nnU-Net 2d 5-fold ensemble (seg v3), vision-agent QC and the frozen brain table. Body volume, soft tissue, colony size and the phylogeny enter the preregistered PGLS model. In the previous draft this schematic was Fig. 1a; Fig. 1 now shows the same steps on real scans.

![](figures/figS2_tk_heldout.png){width=100%}

**Figure S2. seg v3 against human labels on held-out Toulkeridou cases.** Four of the 117 human-labelled stacks of Toulkeridou et al. (2023) that were held out in cross-validation. *Selection rule:* the cases nearest the 10th, 50th and 90th percentile of per-case held-out Dice (a–c) and the case with the lowest Dice (d–e). Each case was predicted by the seg v3 model of the fold whose validation set contains it; folds were split by species, so the species was not in training. Predictions were re-run on CPU from the stored checkpoints with the nnU-Net validation settings (2d, mirroring, tile step 0.5). Blue: human label; orange, dashed: seg v3.

- **(a–d)** The slice with the largest labelled area. Titles give the held-out Dice and volume error stored by the training run (full volume); tiles give the Dice of the shown slice.
- **(e)** For the lowest case, the slice with the largest label–prediction disagreement in the full-volume prediction.

Full volumes were re-predicted for two cases and reproduce the stored values (Dice 0.9806 and 0.9806 for TK_057, 0.4951 and 0.4952 for TK_063). In the lowest case, the human label of this re-sliced stack appears displaced from the visible brain tissue: it covers muscle and air spaces and misses part of the neuropil that seg v3 outlines. The other stack of the same species (TK_062, sliced along another axis) has a held-out Dice of 0.969. This low value may therefore reflect a misaligned reference rather than a segmentation failure; the case is counted unchanged in Fig. 1d–e and Table S2. Across the 117 cases the median Dice was 0.969, and 3 cases were below 0.90. The dataset authors cropped and resampled the images to 520 × 520 pixels per slice, so no physical scale is given.

*Data:* images and labels in `runs/20261113-181636_*/` (`outputs/nnUNet_raw/Dataset501_AntBrainTK/`); checkpoints and stored metrics in `runs/20261114-212341_*/` and `runs/20261114-215207_*/` (`outputs/fold_k/`); displayed slices and predictions in `tk_heldout_slices.npz` (`preprint/figures/image_data/`).

![](figures/figS3_cuticle_qc.png){width=100%}

**Figure S3. Body and cuticle segmentation by the owners' program, and the QC rule.** One specimen accepted and one rejected by the adaptive rule (Section 2.3). *Selection rule:* drawn at random (seed 20261116) from the 92 specimens accepted at threshold 110 and from the 21 specimens rejected with body(110)/body(130) > 1.10, the signature of a bright mounting medium entering the mask. These 21 are among the 37 specimens for which neither threshold gave a stable body volume; 7 further specimens failed the cuticle/body limits. Left: the QC sheet written by our run of the owners' program, with its tiles rearranged. The top row shows four evenly spaced slices of the processed range, and the bottom row the segmentation at threshold 110 (cuticle green, soft tissue blue). Right: body cross-section of every sampled slice (every third slice) at thresholds 110, 130 and 145, with the specimen's body volume at each threshold; grey lines mark the slices shown on the left.

- **(a)** Accepted: the three thresholds give the same body (body(110)/body(130) = 1.01).
- **(b)** Rejected: at threshold 110 the mounting medium fills the vial over long stretches of the stack, so body(110) is 15.6 times body(130); threshold 145 also differs from 130 by more than 10%.

*Data:* volumes and rule inputs in `analysis/r3_extension_v1/cuticle_final.jsonl`; QC sheets and per-slice areas in `runs/20261115-172355_*/` (`outputs/qc/<key>.jpg`, `outputs/slices/<key>.npz`).

![](figures/figS4_qc_rejections.png){width=92%}

**Figure S4. Masks rejected by the vision-agent QC.** Three masks from the final QC round on seg v3 ensemble masks in boxes from the final finder: 268 masks were reviewed, 124 accepted and 144 rejected. The rejected masks fall into three categories by the agent's box verdict: box not on the head (62), box position unsure (45) and box on the head but mask not good or brain not fully in the box (37). *Selection rule:* within each category, one mask was drawn at random (seed 20261116) among specimens absent from the frozen table and with a non-empty mask (20, 25 and 28 eligible). Left: the contact sheet exactly as the agent reviewed it: eight slices through the head box with the mask outline in red. Right: the verdict fields and the agent's note, verbatim.

- **(a)** The agent judged the masked structure to be the crop or gut, not the brain.
- **(b)** The agent could not decide whether the box was on the head, and judged the mask incomplete.
- **(c)** The agent judged the box to be on the head but the brain to be cut off by a box face.

*Data:* verdicts in `meta/brains_v5_auto_review/`; contact sheets in `review/brains_v5/qc/<key>.jpg`, identical by SHA-256 to `outputs/qc/<key>.jpg` of the segmentation run `runs/20261115-065118_*/` or `runs/20261115-065120_*/`.

## References

Aiello LC, Wheeler P (1995) The expensive-tissue hypothesis: the brain and the digestive system in human and primate evolution. *Current Anthropology* 36(2): 199–221. doi:10.1086/204350

Bell-Roberts L, Turner JFR, Werner GDA, Downing PA, Ross L, West SA (2024) Larger colony sizes favoured the evolution of more worker castes in ants. *Nature Ecology & Evolution* 8(10): 1959–1971. doi:10.1038/s41559-024-02512-7

Borowiec ML, Zhang YM, Neves K, Ramalho MO, Fisher BL, Lucky A, Moreau CS (2025) Evaluating UCE data adequacy and integrating uncertainty in a comprehensive phylogeny of ants. *Systematic Biology* 74(5): 700–722. doi:10.1093/sysbio/syaf001 (data: Zenodo doi:10.5281/zenodo.14338504)

Burchill AT, Moreau CS (2016) Colony size evolution in ants: macroevolutionary trends. *Insectes Sociaux* 63(2): 291–298. doi:10.1007/s00040-016-0465-3

Bürkner P-C (2017) brms: an R package for Bayesian multilevel models using Stan. *Journal of Statistical Software* 80(1): 1–28. doi:10.18637/jss.v080.i01

Carpenter B, Gelman A, Hoffman MD, Lee D, et al. (2017) Stan: a probabilistic programming language. *Journal of Statistical Software* 76(1). doi:10.18637/jss.v076.i01

Dornhaus A, Powell S, Bengston S (2012) Group size and its effects on collective organization. *Annual Review of Entomology* 57: 123–141. doi:10.1146/annurev-ento-120710-100604

Dunbar RIM (1998) The social brain hypothesis. *Evolutionary Anthropology* 6(5): 178–190. doi:10.1002/(SICI)1520-6505(1998)6:5<178::AID-EVAN5>3.0.CO;2-8

Economo EP, Narula N, Friedman NR, Weiser MD, Guénard B (2018) Macroecology and macroevolution of the latitudinal diversity gradient in ants. *Nature Communications* 9: 1778. doi:10.1038/s41467-018-04218-4 (data: Dryad doi:10.5061/dryad.g579t7k)

Feinerman O, Traniello JFA (2016) Social complexity, diet, and brain evolution: modeling the effects of colony size, worker size, brain size, and foraging behavior on colony fitness in ants. *Behavioral Ecology and Sociobiology* 70(7): 1063–1074. doi:10.1007/s00265-015-2035-5

Freckleton RP, Harvey PH, Pagel M (2002) Phylogenetic analysis and comparative data: a test and review of evidence. *The American Naturalist* 160(6): 712–726. doi:10.1086/343873

Godfrey RK, Gronenberg W (2020) Linking colony size with foraging behavior and brain investment in odorous ants (Formicidae: Dolichoderinae). *Brain, Behavior and Evolution* 95(1): 15–24. doi:10.1159/000504643

Gronenberg W, Riveros AJ (2009) Social brains and behavior: past and present. In: Gadau J, Fewell J (eds) *Organization of Insect Societies: From Genome to Sociocomplexity*. Harvard University Press, pp. 377–401. doi:10.4159/9780674272842-021

Hadfield JD (2010) MCMC methods for multi-response generalized linear mixed models: the MCMCglmm R package. *Journal of Statistical Software* 33(2): 1–22. doi:10.18637/jss.v033.i02

Ho LST, Ané C (2014) A linear-time algorithm for Gaussian and non-Gaussian trait evolution models. *Systematic Biology* 63(3): 397–408. doi:10.1093/sysbio/syu005

Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH (2021) nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods* 18(2): 203–211. doi:10.1038/s41592-020-01008-z

Kamhi JF, Gronenberg W, Robson SKA, Traniello JFA (2016) Social complexity influences brain investment and neural operation costs in ants. *Proceedings of the Royal Society B* 283(1841): 20161949. doi:10.1098/rspb.2016.1949

Kamhi JF, Ilieş I, Traniello JFA (2019) Social complexity and brain evolution: comparative analysis of modularity and integration in ant brain organization. *Brain, Behavior and Evolution* 93(1): 4–18. doi:10.1159/000497267

Katzke J, Hita Garcia F, Lösel PD, Azuma F, Faragó T, Aibekova L, Casadei-Ferreira A, Gautam S, et al., The Antscan GAGA Consortium, Baumbach T, Economo EP, van de Kamp T (2026) High-throughput phenomics of global ant biodiversity. *Nature Methods* 23(3): 663–672. doi:10.1038/s41592-026-03005-0

Lakens D (2017) Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science* 8(4): 355–362. doi:10.1177/1948550617697177

Lihoreau M, Latty T, Chittka L (2012) An exploration of the social brain hypothesis in insects. *Frontiers in Physiology* 3: 442. doi:10.3389/fphys.2012.00442

Lösel PD, van de Kamp T, Jayme A, Ershov A, Faragó T, Pichler O, et al. (2020) Introducing Biomedisa as an open-source online platform for biomedical image segmentation. *Nature Communications* 11: 5577. doi:10.1038/s41467-020-19303-w

Marty S, Couto A, Dawson EH, Brard N, d'Ettorre P, Montgomery SH, Sandoz J-C (2025) Ancestral complexity and constrained diversification of the ant olfactory system. *Proceedings of the Royal Society B* 292(2045): 20250662. doi:10.1098/rspb.2025.0662

Matte A, Guénard B, Gautam S, Azuma F, Katzke J, Hita Garcia F, van de Kamp T, Economo EP (2025) The evolution of cheaper workers facilitated larger societies and accelerated diversification in ants. *Science Advances* 11(51): eadx8068. doi:10.1126/sciadv.adx8068 (data and code: Zenodo doi:10.5281/zenodo.15546461)

Niven JE, Laughlin SB (2008) Energy limitation as a selective pressure on the evolution of sensory systems. *Journal of Experimental Biology* 211(11): 1792–1804. doi:10.1242/jeb.017574

O'Donnell S, Bulova SJ, DeLeon S, Khodak P, Miller S, Sulger E (2015) Distributed cognition and social brains: reductions in mushroom body investment accompanied the origins of sociality in wasps (Hymenoptera: Vespidae). *Proceedings of the Royal Society B* 282(1810): 20150791. doi:10.1098/rspb.2015.0791

Pagel M (1999) Inferring the historical patterns of biological evolution. *Nature* 401(6756): 877–884. doi:10.1038/44766

Paradis E, Schliep K (2019) ape 5.0: an environment for modern phylogenetics and evolutionary analyses in R. *Bioinformatics* 35(3): 526–528. doi:10.1093/bioinformatics/bty633

Perl CD, Coto ZN, Johnson R, Johnson M, Graber L, Haas J, Waters JS, Traniello J, Harrison JF (2026) Evolution of neurometabolic frugality in harvester ants. *Proceedings of the Royal Society B* 293(2081): 20260387. doi:10.1098/rspb.2026.0387

Riveros AJ, Seid MA, Wcislo WT (2012) Evolution of brain size in class-based societies of fungus-growing ants (Attini). *Animal Behaviour* 83(4): 1043–1049. doi:10.1016/j.anbehav.2012.01.032

Seid MA, Castillo A, Wcislo WT (2011) The allometry of brain miniaturization in ants. *Brain, Behavior and Evolution* 77(1): 5–13. doi:10.1159/000322530

Toulkeridou E, Gutierrez CE, Baum D, Doya K, Economo EP (2023) Automated segmentation of insect anatomy from micro-CT images using deep learning. *Natural Sciences* 3(4): e20230010. doi:10.1002/ntls.20230010 (data: Dryad doi:10.5061/dryad.qz612jmgv)

Wehner R, Fukushi T, Isler K (2007) On being small: brain allometry in ants. *Brain, Behavior and Evolution* 69(3): 220–228. doi:10.1159/000097057

Software without a journal reference: nlme (Pinheiro J, Bates D, R Core Team; R package 3.1-171), phylolm 2.6.5, cmdstanr, R 4.6.2, Python 3.13, nnU-Net v2. **[CITATION NEEDED: formal software citations if the journal requires them]**

## Appendix A. Deviations from the preregistration

The protocol, including its deviations, is described here.

**Declared at the freeze (prereg v1.0).**
1. Segmentation uses the seg v3 ensemble (Toulkeridou labels + agent-approved AntScan brains) instead of a model trained on Toulkeridou labels only.
2. Mask QC is done by vision agents instead of human review.
3. Body volume uses v8 instead of v6. The v6 slice spacing was wrong for compressed stacks; the median ratio to the owner-measured volumes (S1) changed from 1.15 to 1.04.
4. Registration is a timestamped GitHub commit and tag instead of OSF. The repository is private until the preprint is posted.
5. The measurement-error model was run after the primary test.

**After unblinding.**
6. **Measurement-error model, run length.** It used 4 × 4000 iterations instead of 4 × 2000. With 2000 iterations, one variance component had bulk ESS 372 (< 400). The 2000-iteration estimate is +0.0375 (0.0193–0.0554).
7. **Measurement-error model, m.** m is the held-out error only. The per-specimen between-fold spread named in the preregistration was not recorded by the ensemble.
8. **Measurement-error model, backend.** cmdstanr was used instead of rstan (compilation failure); model and priors unchanged.
9. **R2 species pool.** 272 species (270 + 2 species placed only in the Borowiec tree); the 270-species version is reported alongside (Table S4).
10. **R8 flags.** Only Bell-Roberts Table 21 was used, as preregistered; unlisted species count as unflagged.
11. **R1 tree count.** R1 used 6 FBD trees (3 crown + 3 stem), because the analysis configuration capped tree sets at 6. The preregistration names 200. The owners'-method variant of R12 averaged the same 6 trees. Both were recomputed on all 200 trees for this draft; the results are unchanged (Table 1).
12. **R10.** Implemented as an approximation: genus medians of the species-level S3 means in our species table. The planned extension to ~746 species without species-level colony size was not built.
13. **Not run.** The head-volume variant of R3 (no head masks were produced), and the three secondary analyses of prereg Section 9 (number of worker castes, number of queens, path analysis).
14. **R3 extension.** Measurement protocol and QC rule were specified after unblinding. The QC rule was fixed before the final fits and revised once, on segmentation diagnostics. An interim estimate on 28 added species (+0.007) had been seen before the rule was written. Both rule versions are reported.
15. **Brain definition.** The preregistration's TODO to record the exact Toulkeridou label definition is still open (Section 2.2).

ESS is the effective sample size of the sampler.

## Appendix B. Implementation details

**Scan stacks.** Low-resolution copies were cached for 811 stacks. The true number of slices of every full-resolution stack was read from the first image file directory of the tagged image file format (TIFF); a size-based estimate had been wrong for 551 compressed stacks.

**Frozen brain table.** The table was frozen before unblinding (SHA-256 `df1902432f4680b295c1e68cc68804c6d42a71a445d0dbd2003f90bbaf646ead`).

**Registration.** The analysis plan and outcome rules were preregistered as `prereg_draft.md` v0.1 (13 November 2026). The plan was frozen, with the deviations listed in Appendix A, in commit `3b1e350` and tag `ant-twin-prereg-v1.0` (commit `c01e2ee`, wording fix) on 15 November 2026.

**R2 details.**

- The Borowiec tree has one exemplar per genus. Each of the 100 resamples (seed 2025) attaches one random species per genus to that genus' tip.
- Main dating scheme: root prior 129–158 Ma with independent rates. The other three schemes are reported as sensitivity.
- The species pool was all 272 species with brain, body and colony size; two of them are absent from the Economo tree. 117 of 125 genera matched.

**R8 details.**

- 19 of the 270 primary species were flagged: 14 supercolonial, 4 social parasites, 1 both.
- Species outside the Bell-Roberts set were treated as unflagged.

**Measurement-error model** (specimen level; brms 2.23 with CmdStan 2.37):

`log10_brain | se(m, sigma = TRUE) ~ log10_body + log10_colony + iodine + step1.22 + step6.11 + (1 | gr(phylo, cov = A)) + (1 | species)`

- Data: 442 specimens of the 270 species. A is the MCC correlation matrix.
- Known measurement SD: m = log10(1 + 0.05/0.6745) = 0.0311, from the held-out AntScan volume error. Sensitivity: m = 0.0127, from the human-labelled folds.
- Priors as preregistered: b ~ N(0, 0.5), intercept ~ t₃(0, 2.5), SDs ~ half-t₃(0, 0.5).
- Sampling: 4 chains × 4000 iterations (Appendix A, item 6).
- An MCMCglmm fit (4 × 260,000 iterations) served as a cross-check.
