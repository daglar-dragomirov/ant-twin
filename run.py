"""Fixed orx run command for every node: `python -I run.py` (run from the repo root).
What a node does is decided ONLY by committed code + configs/experiment.json on its branch."""
import sys, json, pathlib, importlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
import runlog

cfg = json.loads(pathlib.Path("configs/experiment.json").read_text(encoding="utf-8"))
with runlog.Run(cfg["stage"]) as run:
    print("[config]", json.dumps(cfg, ensure_ascii=False))
    stage = importlib.import_module("stages." + cfg["stage"])
    stage.main(cfg, run.out)
