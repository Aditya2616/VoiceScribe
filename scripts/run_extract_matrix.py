"""Reproduce every extraction number in the README: each extractor on each split, written to results/.

Splits: dev (4 meetings, rules developed here), test (4, looked at while debugging), test2 (3) and held (6, written
after the rules were frozen), plus heldout = test2 + held (9). Model rows use the GPU in float16 when one is
available, else the CPU in float32; device and dtype are recorded in every file.

    python scripts/run_extract_matrix.py [--models 0.5B 1.5B]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicescribe.eval import extract_eval  # noqa: E402
from voicescribe.extract.base import RulesExtractor  # noqa: E402
from voicescribe.extract.local import LocalChooser, LocalExtractor  # noqa: E402

SPLITS = {"dev": "dev", "test": "test", "test2": "test2", "held": "held", "heldout": "test2,held"}


def run(name: str, extractor, meta: dict, out: Path) -> None:
    for label, spec in SPLITS.items():
        ms = extract_eval.load_all(ROOT / "data" / "eval" / "extract", spec)
        r = extract_eval.score(ms, extractor)
        r.update(meta, extractor=name, split=label)
        (out / f"extract_{name}_{label}.json").write_text(json.dumps(r, indent=2), encoding="utf8")
        a, d = r["action_items"], r["decisions"]
        print(f"{name:<22}{label:<8}m={r['meetings']:<2} action P/R/F1 {a['precision']:.2f}/{a['recall']:.2f}/{a['f1']:.2f}"
              f"  decision F1 {d['f1']:.2f}  owners {r['owners']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["0.5B", "1.5B"])
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    run("rules", RulesExtractor(), {}, out)
    for size in args.models:
        for mode, kw in (("owners", {"owners": True}), ("verify", {"owners": False, "verify": True})):
            chooser = LocalChooser(f"Qwen/Qwen2.5-{size}-Instruct")
            ex = LocalExtractor(choose=chooser, **kw)
            run(f"{mode}_{size}", ex, {}, out)
            # device and dtype are known once the model has loaded
            for p in out.glob(f"extract_{mode}_{size}_*.json"):
                r = json.loads(p.read_text(encoding="utf8"))
                r.update(model=chooser.model_name, device=chooser.device, dtype=chooser.dtype)
                p.write_text(json.dumps(r, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
