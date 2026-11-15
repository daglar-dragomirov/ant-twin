"""Stage brain_freeze: one frozen brain record per specimen for the preregistered analysis.

Inputs (all in R2):
* items_key: the 530 approved specimens of the first rounds as [key, box_prefix, ref_mask_prefix]; their final masks come
  from the seg v3 ensemble runs in base_runs (each record carries dice/vol ratio vs the agent-approved reference).
* groups: later QC rounds, applied in order; each has review files and the seg-final runs holding the masks. A key
  approved there (mask good, box on head, brain not cut off) adds or replaces the specimen (grown boxes supersede
  cut-off ones; secondary-candidate boxes add specimens whose first box missed).
Record key = specimen key (folder__vial; the @A/@B/@C suffix of the box is kept in box_key).
Outputs: brains.jsonl (brain_analysis format), review_final.json (the approving verdict per specimen), table.csv,
summary.json with the SHA-256 of brains.jsonl.
Config: items_key, base_runs, groups [{name, reviews, runs}].
"""
import json, hashlib, collections
import r2


def main(cfg, out):
    c = r2.client(); get = lambda k: c.get_object(Bucket=r2.BUCKET, Key=k)["Body"].read()
    keys = r2.list_keys(c, r2.BUCKET, "ant-twin/runs/")

    def records(runs):
        rec = {}
        for run in runs:
            for name in ("brains.jsonl", "brains_at_cancel.jsonl"):
                k = f"ant-twin/runs/{run}/outputs/{name}"
                if k in keys:
                    for l in get(k).decode().splitlines():
                        if l.strip(): x = json.loads(l); x["seg_run"] = run; rec[x["key"]] = x
        return rec

    ok = lambda r: r.get("mask") == "good" and r.get("box_on_head") is True and r.get("brain_fully_in_box") is not False
    final, verdict = {}, {}
    base = records(cfg["base_runs"])
    for k, bp, mp in json.loads(get(cfg["items_key"])):
        x = base[k]; s = k.split("@")[0]
        final[s] = {**x, "box_key": k, "box_prefix": bp, "source": "approved_" + bp.rstrip("/").split("/")[-1]}
        verdict[s] = {"key": s, "mask": "good", "box_on_head": True, "brain_fully_in_box": True,
                      "basis": "agent-approved earlier mask; ensemble agreement", "dice_vs_ref": x.get("dice_vs_ref"), "vol_ratio_vs_ref": x.get("vol_ratio_vs_ref")}
    for g in cfg["groups"]:
        rec = records(g["runs"]); rv = {}
        for f in g["reviews"]:
            for r in json.loads(get(f)): rv[r["key"]] = r
        for k, r in sorted(rv.items()):
            if not ok(r) or k not in rec: continue
            s = k.split("@")[0]
            final[s] = {**rec[k], "box_key": k, "source": g["name"]}
            verdict[s] = {**{kk: r.get(kk) for kk in ("mask", "box_on_head", "brain_fully_in_box", "est_vol_error_pct", "confidence", "note")}, "key": s, "basis": g["name"]}
    lines = []
    for s in sorted(final):
        x = final[s]
        lines.append(json.dumps({"key": s, "box_key": x["box_key"], "source": x["source"], "brain_um3": x["brain_um3"], "brain_vox": x.get("brain_vox"),
                                 "voxel_um": x["voxel_um"], "touches_edge": x.get("touches_edge"), "n_components": x.get("n_components"),
                                 "largest_frac": x.get("largest_frac"), "box_frac": x.get("box_frac"), "seg_run": x["seg_run"]}))
    body = ("\n".join(lines) + "\n").encode()
    (out / "brains.jsonl").write_bytes(body)
    (out / "review_final.json").write_text(json.dumps([verdict[s] for s in sorted(verdict)], indent=0))
    summ = {"specimens": len(final), "sources": dict(collections.Counter(x["source"] for x in final.values())),
            "sha256_brains_jsonl": hashlib.sha256(body).hexdigest()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1)); print("summary", summ, flush=True)
