"""Smoke test of the run/logging contract: writes a small output and sleeps briefly."""
import time, json
def main(cfg, out):
    for i in range(3):
        print("smoke step", i); time.sleep(1)
    (out / "smoke.json").write_text(json.dumps({"ok": True, "cfg": cfg}), encoding="utf-8")
