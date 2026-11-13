"""Cloudflare R2 (S3) helper for the doctorate logs. Credentials are read at runtime from the user's own file;
nothing secret is printed or written anywhere.
Usage: python -I r2.py list-buckets | ensure-bucket <name> | put <bucket> <key> <localfile> | ls <bucket> [prefix] | put-text <bucket> <key> <text>
"""
import sys, re, pathlib
import boto3

CRED = pathlib.Path(r"C:\yandex.disk-daglar.dragomirov\Yandex.Disk\16_Доступы_и_безопасность\Cloudflare_R2_S3\Cloudflare_R2_S3_доступы — MAIN.txt")


def creds():
    lines = [l.strip() for l in CRED.read_text(encoding="utf-8").splitlines()]
    def after(label):
        i = lines.index(label)
        return next(l for l in lines[i + 1:] if l and l != "Click to copy")
    ak, sk = after("Access Key ID"), after("Secret Access Key")
    ep = next(l for l in lines if re.match(r"^https://[0-9a-f]+\.r2\.cloudflarestorage\.com$", l))
    return ak, sk, ep


def client():
    ak, sk, ep = creds()
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=ak, aws_secret_access_key=sk, region_name="auto")


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
    elif cmd == "put-text":
        c.put_object(Bucket=sys.argv[2], Key=sys.argv[3], Body=sys.argv[4].encode("utf-8")); print("put", sys.argv[3])
    elif cmd == "ls":
        pre = sys.argv[3] if len(sys.argv) > 3 else ""
        for o in c.list_objects_v2(Bucket=sys.argv[2], Prefix=pre).get("Contents", []): print(o["Size"], o["Key"])
