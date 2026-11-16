"""Build manuscript_v1.pdf: insert figure images above their legends, convert with pandoc (pypandoc_binary) + xelatex.

Main figures sit above "**Figure N.**" in the Figure legends section, supplementary figures above "**Figure SN.**" in the
Supplementary figures section. Figure 1 comes from figures/make_image_figures.py, Fig. S1 and Figs 2-4 from
figures/make_figures.py, Figs S2-S4 from make_image_figures.py.
"""
import re, pathlib, pypandoc
P = pathlib.Path(__file__).resolve().parent.parent
s = (P / "manuscript_v1.md").read_text(encoding="utf-8")
figs = {"1": ("fig1_scan_to_brain.png", 96), "2": ("fig2_allometry_colony.png", 100), "3": ("fig3_forest_beta2.png", 100),
        "4": ("fig4_denominators_cuticle.png", 100), "S1": ("figS1_pipeline_schematic.png", 100),
        "S2": ("figS2_tk_heldout.png", 100), "S3": ("figS3_cuticle_qc.png", 100), "S4": ("figS4_qc_rejections.png", 92)}
for n, (f, w) in figs.items():
    assert (P / "figures" / f).exists(), f
    s, k = re.subn(rf"^(\*\*Figure {n}\.)", rf"![](figures/{f}){{width={w}%}}\n\n\1", s, count=1, flags=re.M)
    assert k == 1, f"legend for Figure {n} not found"
# pandoc needs a blank line before a list that follows a paragraph line (e.g. "**Figure 1. ...**" then "- **(a)**");
# without it the items run on inline as "- (a) ..."
s, nl = re.subn(r"(?m)^((?![ \t]*[-|*+]\s)[^\n]*\S[^\n]*)\n(- )", r"\1\n\n\2", s)
print("blank lines inserted before lists:", nl)
(P / "pdf" / "manuscript_v1_pdf.md").write_text(s, encoding="utf-8")
pypandoc.convert_file(str(P / "pdf" / "manuscript_v1_pdf.md"), "pdf", outputfile=str(P / "pdf" / "manuscript_v1.pdf"),
    extra_args=["--pdf-engine=xelatex", f"--resource-path={P}", "-V", "geometry:margin=2.2cm", "-V", "mainfont=Cambria",
                "-V", "fontsize=10pt", "-V", "colorlinks=true", "--toc", "-V", "toc-title=Contents"])
print("ok", P / "pdf" / "manuscript_v1.pdf")
