# ANT-TWIN — primary preregistered test, v1 (15 Nov 2026)

Run: orx node `73e0da9c`; R2 `ant-twin/runs/20261115-100746_brain-analysis-v1-primary-unblinded-preregistere_DESKTOP-7Q7S0N3/`. The preregistration was frozen beforehand (tag `ant-twin-prereg-v1.0`, commit `c01e2ee`).

**Inputs.** Frozen brain table SHA-256 `df1902…6ead` (689 specimens). Body volume: S1 > v8. 270 species with brain, body and colony size in the tree.

## Primary result (preregistered decision rules)

Model: log10 brain ~ log10 body + log10 colony + iodine + scan-step class; PGLS-λ on the Economo 2018 MCC tree.

| | β₂ (log10 colony) | 95% CI | p | λ |
|---|---|---|---|---|
| MCC tree, n = 270 | **+0.035** | [+0.017; +0.054] | 0.0002 | 0.72 |
| 100 posterior trees (pooled) | median +0.033 | [+0.014; +0.051] | all p < 0.05 | 0.64 |

**Outcome: support H2 (robust over trees).** In larger colonies, workers have relatively *larger* brains for their body size. That is about +8% brain volume per tenfold increase in colony size. The 90% CI is [0.020; 0.051]. **H1 is rejected:** "smaller brains in larger colonies" is not supported, because no tree or check gives a negative estimate.

The Haller gate passed: brain ~ body slope 0.489 [0.447; 0.530], R² = 0.85.

## Robustness (preregistered R1–R14)

The same outcome (support H2) holds in:
- **R1:** FBD trees, +0.032;
- **R4:** leave-one-subfamily-out, +0.025 to +0.042;
- **R5:** unstained only, +0.035;
- **R6:** all workers, +0.033;
- **R7:** without qualified names, +0.037;
- **R9:** alternative colony measures, +0.027 / +0.040;
- **R10, R11, R13:** +0.036 to +0.041;
- **R12:** λ = 1, +0.040;
- **R14:** +0.027.

**Exception — R3, soft-tissue denominator.** Owners' S1 body minus cuticle, n = 154: β₂ = +0.006 [−0.013; +0.024], p = 0.56, outcome undecided.

**Not run:**
- R2: Borowiec 2025 tree not yet in R2;
- R8: supercolonial / social-parasite flags not yet joined;
- brms measurement-error model: pending.

## Exploratory decomposition of R3 (not preregistered; reported as such)

| Same 154 species | β₂ | 95% CI |
|---|---|---|
| Body denominator | +0.020 | [+0.002; +0.038], p = 0.03 |
| Soft-tissue denominator | +0.006 | [−0.013; +0.024] |

In this subset, about two thirds of the colony effect comes from the denominator. Species with larger colonies have a relatively thinner cuticle: corr(log10 colony, log10 soft/body) = +0.29. This matches Matte et al. 2025.

## Reading (strongest claim the data support)

1. Worker brains do **not** shrink with colony size. Across 270 species the relative brain size is maintained or increases (β₂ > 0 in every analysis).
2. Relative to soft tissue, brain investment is roughly constant (β₂ ≈ 0, n = 154). Most of the positive association with total body size reflects thinner cuticle in large-colony species.
3. So the economy in large colonies falls on the protective cuticle, not on neural tissue.

**Next.** Implement R2 and R8. Run the brms check. Measure head volume: R3 needs head masks. Extend the soft-tissue denominator beyond the 154 species with S1 data, using a cuticle segmentation of our own.

## Correction (16 Nov 2026)

The exploratory "two thirds of the effect comes from the denominator" reading is not supported like-for-like (see `r3_extension_v1.md`, Correction): with body measured by the same instrument as soft tissue, β₂ ≈ +0.012–0.014 (CI includes 0). The preregistered outcome label is unchanged, but its magnitude depends on the body-volume source (S1-only bodies: +0.010 [−0.011; 0.031], n = 155). Robust conclusion: worker brains do not shrink with colony size.
