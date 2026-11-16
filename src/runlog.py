"""Run logging for ANT-TWIN: one folder per run, periodic sync to R2, final verified upload, global RUNLOG.jsonl.

Layout in R2 (bucket dsc-biology):
  ant-twin/runs/<run_id>/meta.json      experiment/branch/commit/host/command/stage/start/end/status/r2_verified
  ant-twin/runs/<run_id>/stdout.log     full console output (tee)
  ant-twin/runs/<run_id>/env.txt        python, packages, GPU
  ant-twin/runs/<run_id>/outputs/...    everything the stage wrote
  ant-twin/RUNLOG.jsonl                 one line per run, appended at start and finished at end (chronological index)
run_id = <UTC yyyymmdd-HHMMSS>_<branch>_<host>
"""
import os, sys, io, json, time, socket, platform, subprocess, threading, datetime, pathlib, traceback
import r2

PREFIX = "ant-twin"
SYNC_EVERY = 300  # seconds


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<{e}>"


class Tee(io.TextIOBase):
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
        return len(s)
    def flush(self):
        for st in self.streams:
            if not st.closed: st.flush()


class Run:
    def __init__(self, stage, root="outputs"):
        now = datetime.datetime.now(datetime.timezone.utc)
        self.branch = os.environ.get("ORX_EXPERIMENT") or sh("git rev-parse --abbrev-ref HEAD") or "nobranch"
        self.commit = os.environ.get("ORX_COMMIT") or sh("git rev-parse HEAD")
        self.host = os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
        safe = lambda s: "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in s)[:60]
        self.run_id = f"{now:%Y%m%d-%H%M%S}_{safe(self.branch)}_{safe(self.host)}"
        self.dir = pathlib.Path(root) / self.run_id; (self.dir / "outputs").mkdir(parents=True, exist_ok=True)
        self.key = f"{PREFIX}/runs/{self.run_id}"
        self.meta = {"run_id": self.run_id, "stage": stage, "branch": self.branch, "commit": self.commit,
                     "orx_run_id": os.environ.get("ORX_RUN_ID"), "orx_experiment": os.environ.get("ORX_EXPERIMENT"), "orx_title": os.environ.get("ORX_EXPERIMENT_TITLE"), "orx_parent": os.environ.get("ORX_PARENT_EXPERIMENT"), "host": self.host, "command": " ".join(sys.argv),
                     "start_utc": now.isoformat(), "end_utc": None, "status": "running", "r2_prefix": self.key,
                     "r2_verified": False, "last_sync_utc": None}
        self.c = r2.client(); self._stop = threading.Event(); self._lock = threading.Lock()

    @property
    def out(self): return self.dir / "outputs"

    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _runlog_append(self, rec):
        key = f"{PREFIX}/RUNLOG.jsonl"
        try:
            body = self.c.get_object(Bucket=r2.BUCKET, Key=key)["Body"].read().decode("utf-8")
        except Exception:
            body = ""
        body += json.dumps(rec, ensure_ascii=False) + "\n"
        self.c.put_object(Bucket=r2.BUCKET, Key=key, Body=body.encode("utf-8"))

    def sync(self):
        with self._lock:
            self.meta["last_sync_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(); self._write_meta()
            remote = r2.list_keys(self.c, r2.BUCKET, self.key + "/")
            for p in self.dir.rglob("*"):
                if p.is_file():
                    k = f"{self.key}/{p.relative_to(self.dir).as_posix()}"
                    if remote.get(k) != p.stat().st_size or p.name in ("stdout.log", "meta.json"):
                        self.c.upload_file(str(p), r2.BUCKET, k)

    def verify(self):
        remote = r2.list_keys(self.c, r2.BUCKET, self.key + "/")
        local = {f"{self.key}/{p.relative_to(self.dir).as_posix()}": p.stat().st_size for p in self.dir.rglob("*") if p.is_file()}
        missing = [k for k, s in local.items() if remote.get(k) != s and not k.endswith(("stdout.log", "meta.json"))]
        return not missing, missing

    def _loop(self):
        while not self._stop.wait(SYNC_EVERY):
            try: self.sync()
            except Exception as e: print(f"[runlog] periodic sync failed: {e}", file=sys.__stderr__)

    def __enter__(self):
        self.log = open(self.dir / "stdout.log", "a", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, self.log); sys.stderr = Tee(sys.__stderr__, self.log)
        (self.dir / "env.txt").write_text("\n".join([platform.platform(), sys.version, sh("pip freeze 2>/dev/null || uv pip freeze"),
                                                      sh("nvidia-smi -L")]), encoding="utf-8")
        self._write_meta(); self.sync()
        self._runlog_append({k: self.meta[k] for k in ("run_id", "stage", "branch", "commit", "host", "start_utc", "status", "r2_prefix")})
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[runlog] run {self.run_id} -> r2://{r2.BUCKET}/{self.key}/ (sync every {SYNC_EVERY}s)")
        return self

    def __exit__(self, et, ev, tb):
        if et: traceback.print_exception(et, ev, tb)
        self._stop.set()
        self.meta["status"] = "failed" if et else "done"
        self.meta["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__; self.log.close()
        self.sync(); ok, missing = self.verify()
        self.meta["r2_verified"] = ok; self.meta["r2_missing"] = missing[:50]; self.sync()
        self._runlog_append({"run_id": self.run_id, "end_utc": self.meta["end_utc"], "status": self.meta["status"], "r2_verified": ok})
        print(f"[runlog] {self.run_id} {self.meta['status']} r2_verified={ok}")
        return False
