"""Tune diarisation on the dev meetings only (synthesised speech with exact turns), then report on the held-out ones.

    python scripts/tune_diarize.py sentences     # the method in use: embed each recognised sentence, cluster
    python scripts/tune_diarize.py segmentation  # the baseline: sherpa-onnx segmentation + clustering pipeline
    python scripts/eval_audio_owners.py      # applies the chosen settings to every split (results/diarize_report.json)

Needs outputs/synth/ from scripts/synth_eval_meetings.py and the diarisation models (voicescribe fetch-models
--diarize). Runs on the CPU with 2 threads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicescribe import audio  # noqa: E402
from voicescribe.diarize import EMBEDDINGS, cluster_spans, diarize, embed_spans  # noqa: E402
from voicescribe.eval import extract_eval  # noqa: E402
from voicescribe.eval.diar_metrics import der, summarise  # noqa: E402
from voicescribe.schema import Word  # noqa: E402
from voicescribe.segment import clean_words, split_sentences  # noqa: E402

SYNTH = ROOT / "outputs" / "synth"
OUT = ROOT / "results"
THREADS = 2
CHOSEN = OUT / "diarize_chosen.json"


def meetings(split: str) -> list[tuple[str, object, list]]:
    out = []
    for m in extract_eval.load_all(ROOT / "data" / "eval" / "extract", split):
        timing = json.loads((SYNTH / f"{m.name}.timing.json").read_text(encoding="utf8"))
        ref = [(s["start"], s["end"], s["speaker"]) for s in timing["sentences"]]
        out.append((m.name, audio.decode(SYNTH / f"{m.name}.wav"), ref))
    return out


def evaluate(data: list, oracle: bool = False, **cfg) -> dict:
    """Diarise every meeting with `cfg`; with oracle=True the true number of speakers is given to the clustering."""
    per, t0 = [], time.perf_counter()
    for name, samples, ref in data:
        k = len({t[2] for t in ref}) if oracle else -1
        hyp = diarize(samples, ROOT / "models", num_speakers=k, threshold=cfg.get("threshold", 0.5),
                      min_on=cfg.get("min_on", 0.3), min_off=cfg.get("min_off", 0.4), int8=cfg.get("int8", False), threads=THREADS,
                      embedding=cfg.get("embedding", "wespeaker_en_voxceleb_resnet34"))  # fmt: skip
        r = der(ref, hyp)
        r["speech"] = sum(e - s for s, e, _ in ref)
        r["meeting"] = name
        per.append(r)
    s = summarise(per)
    s["seconds"] = round(time.perf_counter() - t0, 1)
    return {"config": {**cfg, "speakers_given": oracle}, "summary": s, "per_meeting": per}


def tune_sentences() -> None:
    """Sweep embedding model, clustering threshold and minimum sentence length on the dev meetings (ASR sentences)."""
    dev = []
    for m in extract_eval.load_all(ROOT / "data" / "eval" / "extract", "dev"):
        words = clean_words([Word(**w) for w in json.loads((SYNTH / f"{m.name}.words.small.en.json").read_text(encoding="utf8"))])
        spans = [(s.start, s.end) for s in split_sentences(words)]
        timing = json.loads((SYNTH / f"{m.name}.timing.json").read_text(encoding="utf8"))
        ref = [(t["start"], t["end"], t["speaker"]) for t in timing["sentences"]]
        dev.append((m.name, audio.decode(SYNTH / f"{m.name}.wav"), spans, ref))
    results = []
    for emb_name in EMBEDDINGS:
        embs = [embed_spans(x, spans, ROOT / "models", emb_name, THREADS) for _, x, spans, _ in dev]
        for min_s in (0.5, 1.0, 1.5, 2.0):
            for th in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7):
                for oracle in (False, True) if th == 0.3 else (False,):
                    per = []
                    for (name, _, spans, ref), emb in zip(dev, embs, strict=True):
                        k = len({t[2] for t in ref}) if oracle else None
                        lab = cluster_spans(emb, [b - a for a, b in spans], k, th, min_s)
                        hyp = [(a, b, f"S{c}") for (a, b), c in zip(spans, lab, strict=True)]
                        r = der(ref, hyp)
                        r["speech"] = sum(e - st for st, e, _ in ref)
                        r["meeting"] = name
                        per.append(r)
                    sm = summarise(per)
                    cfg = {"embedding": emb_name, "threshold": th, "min_seconds": min_s, "speakers_given": oracle}
                    results.append({"config": cfg, "summary": sm, "per_meeting": per})
                    print(f"{cfg}: DER {sm['der']:.3f} count-exact {sm['speaker_count_exact']:.2f} MAE {sm['speaker_count_mean_abs_error']:.2f}", flush=True)
    (OUT / "diarize_sentence_tuning_dev.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    choose_sentences()


def choose_sentences() -> None:
    """Pick the dev setting with the lowest DER; among near-ties (within 0.2 points) prefer the one whose neighbouring
    thresholds also score well, so the choice sits on a plateau and not on a lucky spike."""
    results = json.loads((OUT / "diarize_sentence_tuning_dev.json").read_text(encoding="utf8"))
    est = [r for r in results if not r["config"]["speakers_given"]]
    by = {(r["config"]["embedding"], r["config"]["min_seconds"], r["config"]["threshold"]): r["summary"]["der"] for r in est}

    def plateau(r: dict) -> float:
        c = r["config"]
        ths = sorted({k[2] for k in by if k[:2] == (c["embedding"], c["min_seconds"])})
        i = ths.index(c["threshold"])
        if i in (0, len(ths) - 1):
            return 1.0  # an edge of the grid has only one neighbour: not shown to be a plateau
        near = [by[(c["embedding"], c["min_seconds"], t)] for t in ths[i - 1 : i + 2]]
        return sum(near) / len(near)

    top = min(r["summary"]["der"] for r in est)
    ties = [r for r in est if r["summary"]["der"] <= top + 0.002]
    best = min(ties, key=lambda r: (plateau(r), r["summary"]["speaker_count_mean_abs_error"]))
    cfg = {k: v for k, v in best["config"].items() if k != "speakers_given"}
    CHOSEN.write_text(json.dumps({"method": "sentences", "tuned_on": "dev meetings only", "params": cfg, "dev_summary": best["summary"]}, indent=2), encoding="utf8")
    print("chosen:", cfg, round(best["summary"]["der"], 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sentences", "choose", "segmentation"])
    args = ap.parse_args()
    if args.mode == "sentences":
        tune_sentences()
        return
    if args.mode == "choose":
        choose_sentences()
        return
    if args.mode == "segmentation":
        dev = meetings("dev")
        results = []

        def run(**cfg) -> dict:
            r = evaluate(dev, **cfg)
            results.append(r)
            sm = r["summary"]
            print(f"{cfg}: DER {sm['der']:.3f} frame acc {sm['frame_accuracy']:.3f} count-exact {sm['speaker_count_exact']:.2f}", flush=True)
            return r

        key = lambda r: (r["summary"]["der"], r["summary"]["speaker_count_mean_abs_error"])  # noqa: E731
        best = None
        for emb in EMBEDDINGS:
            for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                r = run(embedding=emb, threshold=th)
                if best is None or key(r) < key(best):
                    best = r
        cfg = {k: v for k, v in best["config"].items() if k != "speakers_given"}
        for on, off in ((0.1, 0.2), (0.1, 0.4), (0.3, 0.2), (0.5, 0.4)):
            r = run(**{**cfg, "min_on": on, "min_off": off})
            if key(r) < key(best):
                best = r
        cfg = {k: v for k, v in best["config"].items() if k != "speakers_given"}
        r = run(**{**cfg, "int8": True})
        if key(r) < key(best):
            best = r
        cfg = {k: v for k, v in best["config"].items() if k != "speakers_given"}
        run(oracle=True, **cfg)
        (OUT / "diarize_segmentation_tuning_dev.json").write_text(json.dumps(results, indent=2), encoding="utf8")
        params = {k: v for k, v in cfg.items()}
        (OUT / "diarize_segmentation_chosen.json").write_text(json.dumps({"tuned_on": "dev meetings only", "params": params, "dev_summary": best["summary"]}, indent=2), encoding="utf8")
        print("chosen:", params)


if __name__ == "__main__":
    main()
