# ANT-TWIN

**Do ants that live in larger colonies have smaller brains?** This repository holds the code, the preregistration,
the analysis and the manuscript of a comparative test of that question across 270 ant species. The brains were
measured in 3D in synchrotron micro-CT scans from [AntScan](https://doi.org/10.1038/s41592-026-03005-0) (Katzke et al. 2026).

The answer is no. Workers of species with larger colonies do not have smaller brains for their body size.

## Main results

| | Value |
|---|---|
| Brains measured | 689 workers of 325 species, automated 3D segmentation |
| Segmentation accuracy on expert labels of species the model never saw | median Dice 0.969 (0.962–0.973 across the five folds), median volume error 2% |
| Preregistered test, colony-size slope β₂ (270 species, PGLS) | +0.035 (95% CI 0.017 to 0.054) |
| Lowest bound of any 95% interval in the main robustness table | −0.015 (a reduction of the preregistered minimal size, −0.02, is excluded) |
| Same test with body size measured identically for all species | +0.014 (−0.004 to 0.031) |
| Same test with soft tissue (body minus cuticle) as the size measure, 235 species | +0.010 (−0.006 to 0.026) |

**In short.** The prediction of "distributed cognition", that individual brains shrink as colonies grow, is not
supported by any analysis. The preregistered estimate is positive, but its size depends on how body volume is
measured; with a uniform body measure it is close to zero. The robust conclusion is therefore that brains do
not shrink; an increase is not established. The manuscript reports both estimates.

## Manuscript

- PDF: [`paper/preprint/source/ant-twin-preprint.pdf`](paper/preprint/source/ant-twin-preprint.pdf)
- LaTeX source (compiles with pdfLaTeX + BibTeX, also on Overleaf): [`paper/preprint/source/`](paper/preprint/source/)
- Figures and their scripts: [`paper/preprint/figures/`](paper/preprint/figures/); tables: [`paper/preprint/tables/`](paper/preprint/tables/)

## Preregistration

The analysis plan, the smallest effect size of interest and the decision rules were fixed on 15 November 2026,
before any model with colony size was fitted to real brain volumes. The frozen files are in
[`paper/prereg/`](paper/prereg/):

| File | SHA-256 |
|---|---|
| `prereg_v1.0_frozen.md` | `212bc9910fc7d992cd1b06e6f278f6a1b665f8c36e1da5cd1b4704b98ec15230` |
| `prereg_draft.md` | `94689629bf5eb1a005d89ad8d96be2665ee208576521e1f664adf84890b62c2e` |

They were committed with the tag `ant-twin-prereg-v1.0` (commit `c01e2ee`) in the author's project repository,
whose history is available to editors and reviewers on request. Every deviation is listed in Appendix A of the
manuscript. The preregistration and some working reports in [`paper/results/`](paper/results/) are written in
Russian, the author's working language; the manuscript and its appendices state the plan, the rules and all
results in English.

## How the measurement works

1. **Specimen table.** AntScan workers are joined to colony-size data (Bell-Roberts et al. 2024; Matte et al. 2025)
   and to the ant phylogeny (Economo et al. 2018). `src/stages/specimen_table.py`
2. **Low-resolution cache and head finder.** A 3D nnU-Net finds the head in a downsampled stack; a box around it is
   cut from the full-resolution scan by HTTP range requests, so whole scans are never downloaded.
   `lowres_cache.py`, `finder_*.py`, `boxes_*.py`, `true_pages.py`, `src/bigtiff_range.py`
3. **Brain segmentation.** A 2D nnU-Net, trained on the expert labels of Toulkeridou et al. (2023) and on AntScan
   brains approved in earlier rounds, segments the brain in each head box; the final mask is a five-fold ensemble.
   `seg_v1.py` (training), `seg_predict.py` (inference)
4. **Quality control.** Every mask is reviewed against a fixed written protocol by a vision-capable language-model
   agent; disputed masks get a second review. The verdicts and review sheets are kept with the data.
5. **Frozen brain table.** `brain_freeze.py` writes the table and its checksum.
6. **Body, cuticle and soft tissue.** Our body masks give one volume (`body_volume.py`). The owners'
   own segmentation program gives body and cuticle volumes (Matte et al. 2025, vendored unchanged in `src/vendor/matte2025/`), validated against
   their published values (`cuticle_matte.py`).
7. **Analysis.** Phylogenetic generalised least squares (PGLS) with Pagel's λ (`src/analysis/pgls.py`,
   `src/stages/brain_analysis.py`), cross-checked against `nlme` and `phylolm`; robustness checks and the
   measurement-error models (brms, MCMCglmm) in [`paper/scripts/robustness_v1/`](paper/scripts/robustness_v1/).

[`STAGES.md`](STAGES.md) lists every stage file with the branch and commit it was taken from.

## How runs are organised

Each experiment lives on its own branch (`orx/<experiment-name>`). A branch differs from the others only in its code
and in `configs/experiment.json`, and every run uses the same command:

```
python -I run.py
```

`run.py` reads the stage name from the config and runs `src/stages/<stage>.py`. Each run writes a `meta.json` with
the commit, the machine, the start and end time and the status, plus its full log and outputs. Runs are stored
in an object store; [`paper/r2_index/INDEX.md`](paper/r2_index/INDEX.md) maps every stage to its accepted run, and
[`RUNS.md`](paper/r2_index/RUNS.md) lists all 157 runs, including failed and superseded ones, with the reason.
A public release with a DOI (Zenodo) will contain the brain table, the masks, the QC verdicts and the analysis
outputs.

## Data sources

| Data | Source | Licence |
|---|---|---|
| Micro-CT scans of 2,193 ants, 792 species | AntScan (Katzke et al. 2026, [10.1038/s41592-026-03005-0](https://doi.org/10.1038/s41592-026-03005-0)), served by biomedisa.info | CC BY 4.0 |
| Expert brain labels (training and held-out tests) | Toulkeridou et al. 2023, Dryad [10.5061/dryad.qz612jmgv](https://doi.org/10.5061/dryad.qz612jmgv) | CC0 |
| Body and cuticle volumes, colony sizes, segmentation program | Matte et al. 2025, Zenodo [10.5281/zenodo.15546461](https://doi.org/10.5281/zenodo.15546461) | CC BY 4.0 |
| Phylogeny (MCC and posterior trees) | Economo et al. 2018, Dryad [10.5061/dryad.g579t7k](https://doi.org/10.5061/dryad.g579t7k) | CC0 |
| Genus-level dated phylogeny | Borowiec et al. 2025, Zenodo [10.5281/zenodo.14338504](https://doi.org/10.5281/zenodo.14338504) | CC BY 4.0 |
| Colony traits and supercolony / parasite flags | Bell-Roberts et al. 2024, [10.1038/s41559-024-02512-7](https://doi.org/10.1038/s41559-024-02512-7) | see source |

## Reproducing a stage

```
pip install -r requirements.txt
git checkout orx/<experiment-name>     # the branch of the run you want to repeat
python -I run.py
```

The scans are read directly from the AntScan server. Stages that read intermediate results expect access to the
object store; until the Zenodo release, those inputs are available from the author on request.

## Use of AI tools

Large-language-model assistants (Claude, Anthropic) helped write the code, run the computations and draft the text,
under the author's direction. Mask quality control by vision agents is part of the method. The manuscript's
*AI use statement* lists what was done and the checks that do not depend on the assistants: accuracy against expert
labels, agreement with the owners' published volumes, independent statistical implementations and a timestamped
preregistration.

## Citation

Dragomirov D. (2026). Workers of large-colony ants do not have smaller brains: a preregistered test. Preprint.
ORCID: [0009-0003-6110-4188](https://orcid.org/0009-0003-6110-4188).
(DOI to be added.)

Contact: open an issue in this repository.
