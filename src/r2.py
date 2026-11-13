"""Cloudflare R2 (S3) helper for the doctorate. Credentials are read at runtime, never printed or committed:
  1) ~/.config/dsc/r2.json  {"access_key": ..., "secret_key": ..., "endpoint": ...}  (used on pods; copied over SSH)
  2) the user's original credentials file on Yandex.Disk (used on the user's PC).
CLI: python -I r2.py list-buckets | ensure-bucket <b> | put <b> <key> <file> | ls <b> [prefix] | export-json <path>
"""
import sys, re, json, pathlib
import boto3

YANDEX = pathlib.Path(r"C:\yandex.disk-daglar.dragomirov\Yandex.Disk\16_Доступы_и_безопасность\Cloudflare_R2_S3\Cloudflare_R2_S3_доступы — MAIN.txt")
LOCAL = pathlib.Path.home() / ".config" / "dsc" / "r2.json"
BUCKET = "dsc-biology"


def creds():
    if LOCAL.exists():
        j = json.loads(LOCAL.read_text(encoding="utf-8"))
        return j["access_key"], j["secret_key"], j["endpoint"]
    lines = [l.strip() for l in YANDEX.read_text(encoding="utf-8").splitlines()]
    def after(label):
        i = lines.index(label)
        return next(l for l in lines[i + 1:] if l and l != "Click to copy")
    ep = next(l for l in lines if re.match(r"^https://[0-9a-f]+\.r2\.cloudflarestorage\.com$", l))
    return after("Access Key ID"), after("Secret Access Key"), ep


def client():
    ak, sk, ep = creds()
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=ak, aws_secret_access_key=sk, region_name="auto")


def list_keys(c, bucket, prefix):
    out, tok = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if tok: kw["ContinuationToken"] = tok
        r = c.list_objects_v2(**kw)
        for o in r.get("Contents", []): out[o["Key"]] = o["Size"]
        if not r.get("IsTruncated"): return out
        tok = r["NextContinuationToken"]


if __name__ == "__main__":
    c, cmd = client(), sys.argv[1]
    if cmd == "list-buckets":
        for b in c.list_buckets().get("Buckets", []): print(b["Name"])
    elif cmd == "ensure-bucket":
        names = [b["Name"] for b in c.list_buckets().get("Buckets", [])]
        if sys.argv[2] not in names: c.create_bucket(Bucket=sys.argv[2]); print("created", sys.argv[2])
        else: print("exists", sys.argv[2])
    elif cmd == "put":
        c.upload_file(sys.argv[4], sys.argv[2], sys.argv[3]); print("put", sys.argv[3])
    elif cmd == "ls":
        for k, s in list_keys(c, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "").items(): print(s, k)
    elif cmd == "export-json":  # write creds for copying to a pod over SSH (file stays outside any repo)
        ak, sk, ep = creds(); p = pathlib.Path(sys.argv[2]); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"access_key": ak, "secret_key": sk, "endpoint": ep}), encoding="utf-8"); print("written", p)
