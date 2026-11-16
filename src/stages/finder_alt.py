"""Stage finder_alt: full-res boxes from the finder's secondary candidates (2nd/3rd largest components).

For each key in the target list (specimens whose main finder box failed vision QC: off-head, gut, empty), every
secondary candidate stored in the finder row becomes a box with the same mapping the finder uses for its main box:
centre and half-size of the low-res bbox times f = (P/n_lowres, H/h_lowres, W/w_lowres), half-size enlarged by
margin_frac, clipped to the stack. Rows are written in finder.jsonl format with keys <key>@C1, <key>@C2.
Config: finder_runs (later runs override earlier ones per key), targets_key, margin_frac.
Outputs: finder.jsonl, summary.json.
"""
import json, collections
import numpy as np
import r2


def main(cfg, out):
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    rows = {}
    for run in cfg["finder_runs"]:
        for l in get(f"ant-twin/runs/{run}/outputs/finder.jsonl").decode().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("status") == "ok": rows[r["key"]] = r
    targets = json.loads(get(cfg["targets_key"])); m = float(cfg.get("margin_frac", 0.6))
    n_out = 0; stat = collections.Counter()
    with open(out / "finder.jsonl", "w", encoding="utf-8") as fo:
        for key in targets:
            r = rows.get(key)
            if r is None: stat["no_row"] += 1; continue
            cands = r.get("candidates", [])[1:]
            if not cands: stat["no_secondary"] += 1; continue
            P, H, W, nlow = r["P"], r["H"], r["W"], r["nlow"]; ls = r["lowres_shape"]
            f = np.array([P / nlow, H / ls[1], W / ls[2]]); lim = np.array([P, H, W])
            for j, cd in enumerate(cands, start=1):
                b = np.array(cd["lowres_bbox"], dtype=float); lo, hi = b[:3], b[3:]
                cc = (lo + hi) / 2 * f; half = (hi - lo) / 2 * f * (1 + m)
                z0, y0, x0 = np.maximum(np.floor(cc - half), 0).astype(int); z1, y1, x1 = np.minimum(np.ceil(cc + half), lim).astype(int)
                rec = {k: r[k] for k in ("species", "species_key", "subfamily", "stained", "voxel_um", "folder", "vial", "url_full", "P", "H", "W", "nlow")}
                rec.update(key=f"{key}@C{j}", base_key=key, status="ok", candidate=j, cand_vox=cd["vox"], lowres_bbox=cd["lowres_bbox"],
                           box_zyx=[int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)])
                fo.write(json.dumps(rec) + "\n"); n_out += 1
            stat["with_secondary"] += 1
    summ = {"targets": len(targets), "rows_out": n_out, **stat}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
