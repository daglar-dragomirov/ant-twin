"""Shared paths/loaders for the robustness_v1 scripts (R8, R2, measurement-error model)."""
import sys, pathlib
sys.path.insert(0, "C:/projects/ant-twin-wt/analysis/src")
import pandas as pd
from analysis import pgls

ROOT = pathlib.Path("C:/projects/dsc-biology/research/ant-twin")
WORK = ROOT / "work" / "robustness"
RUN_OUT = WORK / "run_outputs"   # copy of R2 ant-twin/runs/20261115-100746_brain-analysis-v1-primary-.../outputs/
TD = ROOT / "work/economo2018/inner/Dryad_archive/15k_all_ant_trees"
BR_XLSX = ROOT / "work/bellroberts2024/Supplementary_Tables.xlsx"
SESOI = 0.02


def primary_data():
    return pd.read_csv(RUN_OUT / "primary_data.csv", dtype={"vox_class": str})


def mcc():
    return pgls.read_trees(TD / "15k_NCuniform_stem_mcc.tre")[0]


def posterior(n=None):
    return pgls.read_trees(TD / "15k_NCuniform_stem_posterior.tre", n)
