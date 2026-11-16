"""Single run command for every experiment branch: `python -I run.py` (from the repository root).
What a run does is decided only by the committed code and configs/experiment.json on its branch."""
import sys, json, pathlib, importlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
import runlog

cfg = json.loads(pathlib.Path("configs/experiment.json").read_text(encoding="utf-8"))
with runlog.Run(cfg["stage"]) as run:
    print("[config]", json.dumps(cfg, ensure_ascii=False))
    stage = importlib.import_module("stages." + cfg["stage"])
    stage.main(cfg, run.out)
