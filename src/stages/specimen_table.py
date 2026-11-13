"""Stage specimen_table: one table of every AntScan specimen with download addresses, caste, stain, voxel size and colony size
from two independent sources (AntScan owners, Zenodo 15546461 S3; Bell-Roberts et al. 2024, Supplementary Table 20).
All inputs are fetched inside the run and their sha256 recorded. Outputs: specimens.csv, species.csv, sources.json, summary.json.
"""
import io, json, hashlib, zipfile, urllib.request, re, pathlib
import numpy as np, pandas as pd

UA = {"User-Agent": "dsc-biology-research/0.1 (academic, non-commercial)"}
SRC = {
    "antscan_supp_zip": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12982114/supplementaryFiles",
    "owners_datasets_zip": "https://zenodo.org/records/15546461/files/Datasets.zip?download=1",
    "bell_roberts_xlsx": "https://raw.githubusercontent.com/LouisBell-Roberts/Larger_colony_sizes_favoured_the_evolution_of_more_worker_castes_in_ants/HEAD/Supplementary_Tables/Supplementary_Tables.xlsx",
}
BASE = "https://biomedisa.info/media/antscan/processed"


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
        return r.read()


def key(name):
    """Normalise a species name to genus_species (drop cf./aff./sp. qualifiers, authorities, subspecies)."""
    s = str(name).replace(".", " ").replace("_", " ").lower()
    s = re.sub(r"\b(cf|aff|nr|near|sp|spp|s str|sensu lato)\b", " ", s)
    w = [x for x in re.split(r"\s+", s) if x and re.match(r"^[a-z-]+$", x)]
    return "_".join(w[:2]) if len(w) >= 2 else None


def main(cfg, out):
    raw, sha = {}, {}
    for k, u in SRC.items():
        raw[k] = fetch(u); sha[k] = {"url": u, "bytes": len(raw[k]), "sha256": hashlib.sha256(raw[k]).hexdigest()}
        print("fetched", k, len(raw[k]))
    meta = pd.read_excel(io.BytesIO(zipfile.ZipFile(io.BytesIO(raw["antscan_supp_zip"])).read("41592_2026_3005_MOESM3_ESM.xlsx")), "antscan_metadata")
    oz = zipfile.ZipFile(io.BytesIO(raw["owners_datasets_zip"]))
    def owners(prefix):
        n = next(n for n in oz.namelist() if prefix in n and n.endswith(".xlsx"))
        return pd.read_excel(io.BytesIO(oz.read(n)))
    s1, s3 = owners("Dataset_S1"), owners("Dataset_S3")
    br = pd.read_excel(io.BytesIO(raw["bell_roberts_xlsx"]), "Supplementary Table 20")

    folder = meta["LSDF_parent_folder"].str.extract(r"processed/([^/]+)/")[0]
    vial = meta["vial_label"].astype(str).str.strip()
    t = pd.DataFrame({
        "specimen_code": meta["specimen_code"], "name": meta["Name"], "species_key": meta["Name"].map(key),
        "subfamily": meta["Subfamily"], "genus": meta["Genus"], "caste": meta["caste"], "caste_detail": meta["caste_detail"],
        "stained": meta["stained"], "voxel_um": meta["voxel_size"].astype(str).str.extract(r"([\d.]+)")[0].astype(float),
        "folder": folder, "vial": vial, "doi": meta["doi"], "problems": meta["problems_det"].fillna("").astype(str) + " " + meta["problems_other"].fillna("").astype(str),
        "date_collected": meta["date_collected_start"],
    })
    t["url_full"] = BASE + "/" + t["folder"] + "/" + t["vial"] + ".tif"
    t["url_lowres_slice0"] = BASE + "/" + t["folder"] + "/" + t["vial"] + "_lowres/slice_0000.png"

    # owners' colony sizes: species-level geometric mean of reported means, and max of maxima
    s3k = s3.assign(species_key=(s3["Valid Genus"].astype(str) + " " + s3["Valid Species Name"].astype(str)).map(key))
    s3k["mean"] = pd.to_numeric(s3k["Mean Size"], errors="coerce"); s3k["max"] = pd.to_numeric(s3k["Maximum Size"], errors="coerce")
    g = s3k.dropna(subset=["species_key"]).groupby("species_key")
    own = pd.DataFrame({"colony_mean_owners": g["mean"].apply(lambda v: float(np.exp(np.log(v.dropna()).mean())) if v.notna().any() else np.nan),
                        "colony_max_owners": g["max"].max(), "colony_n_records_owners": g.size()})
    # Bell-Roberts: detect species and colony-size columns by name
    bcols = {c: c for c in br.columns}
    sp_col = next(c for c in br.columns if re.search(r"species|taxon|name", str(c), re.I))
    col_col = next((c for c in br.columns if re.search(r"colony", str(c), re.I)), None)
    caste_col = next((c for c in br.columns if re.search(r"caste", str(c), re.I)), None)
    queen_col = next((c for c in br.columns if re.search(r"queen", str(c), re.I)), None)
    brk = br.assign(species_key=br[sp_col].map(key)).dropna(subset=["species_key"]).drop_duplicates("species_key").set_index("species_key")
    bro = pd.DataFrame(index=brk.index)
    for nm, c in (("colony_br", col_col), ("n_worker_castes_br", caste_col), ("n_queens_br", queen_col)):
        if c is not None: bro[nm] = pd.to_numeric(brk[c], errors="coerce")
    sp = (t.groupby("species_key").agg(subfamily=("subfamily", "first"), n_specimens=("specimen_code", "size"),
                                       n_workers=("caste", lambda c: int((c == "worker").sum())),
                                       n_workers_unstained=("caste", lambda c: 0), n_iodine=("stained", lambda s: int((s == "iodine").sum())))
          .join(own, how="left").join(bro, how="left"))
    sp["n_workers_unstained"] = t[(t.caste == "worker") & (t.stained == "unstained")].groupby("species_key").size().reindex(sp.index).fillna(0).astype(int)
    sp["colony_any"] = sp["colony_mean_owners"].fillna(sp.get("colony_br"))
    t = t.merge(sp[["colony_mean_owners", "colony_max_owners", "colony_any"] + [c for c in ("colony_br", "n_worker_castes_br", "n_queens_br") if c in sp]],
                left_on="species_key", right_index=True, how="left")
    s1k = s1.copy(); s1k.columns = [str(c) for c in s1k.columns]
    t.to_csv(out / "specimens.csv", index=False, encoding="utf-8"); sp.to_csv(out / "species.csv", encoding="utf-8")
    w = t[t.caste == "worker"]
    summary = {
        "specimens": int(len(t)), "workers": int(len(w)), "species_keys": int(t.species_key.nunique()),
        "worker_species_with_colony_owners": int(w.dropna(subset=["colony_mean_owners"]).species_key.nunique()),
        "worker_species_with_colony_any": int(w.dropna(subset=["colony_any"]).species_key.nunique()),
        "worker_specimens_with_colony_any": int(w.colony_any.notna().sum()),
        "worker_species_with_colony_any_unstained": int(w[(w.stained == "unstained")].dropna(subset=["colony_any"]).species_key.nunique()),
        "worker_species_with_colony_any_iodine_only": int(len(set(w[w.stained == "iodine"].dropna(subset=["colony_any"]).species_key) - set(w[w.stained == "unstained"].species_key))),
        "by_folder_workers_with_colony": w.dropna(subset=["colony_any"]).folder.value_counts().to_dict(),
        "bell_roberts_columns_used": {"species": str(sp_col), "colony": str(col_col), "castes": str(caste_col), "queens": str(queen_col)},
        "owners_S1_rows": int(len(s1)),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "sources.json").write_text(json.dumps(sha, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
