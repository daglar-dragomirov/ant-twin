# Vendored: Matte et al. 2025 automated cuticle/body segmentation (unchanged)

Source: Zenodo 10.5281/zenodo.15546461, `Python_scripts.zip` (CC BY 4.0).
Matte A., Guénard B., Azuma F., Katzke J., Gautam S., Hita Garcia F., van de Kamp T., Economo E. P. (2025)
"The evolution of cheaper workers facilitated larger societies and accelerated diversification in ants", Science Advances, adx8068.

Files are byte-identical copies of the authors' `internal__auto_scan_segmentation.py` and
`internal__extract_data_segmentation.py`. Our wrapper (`src/stages/cuticle_matte.py`) only replaces file I/O
(slice reading from an already-open TIFF instead of re-opening the file per slice); the per-slice processing chain
and the per-specimen aggregation (median filter size 25 over slices, then sum) follow the authors' scripts.
