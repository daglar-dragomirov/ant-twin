"""Background checkpoint sync for a pod: every `period` seconds upload nnU-Net training state (checkpoint_latest/best/final,
training logs, plans, splits) from /root/nnunet/res* to R2 ant-twin/checkpoints/<pod>/..., uploading only changed files.
Lets a training resume on another pod (nnUNetv2_train ... --c) after a pod failure.
Usage: python ckpt_sync.py <pod_name> [period_s]
"""
import sys, time, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import r2

pod, period = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600
root = pathlib.Path("/root/nnunet"); c = r2.client(); seen = {}
KEEP = (".pth", ".txt", ".json", ".png", ".pkl")
while True:
    n = 0
    for base in sorted(root.glob("res*")):
        for f in base.rglob("*"):
            if not f.is_file() or f.suffix not in KEEP or "/validation/" in f.as_posix(): continue
            st = (f.stat().st_size, f.stat().st_mtime)
            if seen.get(f) == st or time.time() - st[1] < 20: continue  # skip unchanged or still being written
            key = f"ant-twin/checkpoints/{pod}/{f.relative_to(root).as_posix()}"
            try: c.upload_file(str(f), r2.BUCKET, key); seen[f] = st; n += 1
            except Exception as ex: print("upload failed", key, ex, flush=True)
    sp = pathlib.Path("/root/nnunet/pre")
    for f in sp.glob("*/splits_final.json"):
        key = f"ant-twin/checkpoints/{pod}/pre/{f.relative_to(sp).as_posix()}"
        st = (f.stat().st_size, f.stat().st_mtime)
        if seen.get(f) != st: c.upload_file(str(f), r2.BUCKET, key); seen[f] = st; n += 1
    print(time.strftime("%H:%M:%S"), "uploaded", n, flush=True)
    time.sleep(period)
