"""Owners from audio: rules on text only versus rules plus diarisation, on the synthesised labelled meetings.

    python scripts/eval_audio_owners.py [--whisper small.en] [--threads 2]

Stages (each cached under outputs/synth/): transcribe every meeting once with Whisper, diarise it two ways (the
sentence-level method with the settings in results/diarize_chosen.json, and the sherpa-onnx segmentation pipeline as the
baseline with results/diarize_segmentation_chosen.json), then build each meeting several ways and score the action items against the gold labels
by time overlap with the true sentence timings. Writes results/audio_owners.json. The audio is synthetic speech
with distinct voices, so these numbers are an upper bound for real meetings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicescribe import audio  # noqa: E402
from voicescribe.asr import FasterWhisperTranscriber  # noqa: E402
from voicescribe.diarize import diarize, diarize_sentences  # noqa: E402
from voicescribe.eval import extract_eval  # noqa: E402
from voicescribe.eval.diar_metrics import der, summarise  # noqa: E402
from voicescribe.eval.speech_eval import demo_eval  # noqa: E402
from voicescribe.extract.base import RulesExtractor  # noqa: E402
from voicescribe.pipeline import build_meeting  # noqa: E402
from voicescribe.schema import Word  # noqa: E402
from voicescribe.segment import clean_words, split_sentences  # noqa: E402

SYNTH = ROOT / "outputs" / "synth"
SPLITS = {"dev": "dev", "test": "test", "test2": "test2", "held": "held", "heldout": "test2,held"}
VARIANTS = {
    "text_only": dict(diar=None, people=False),
    "diarised_estimated": dict(diar="sent_est", people=False),
    "diarised_speakers_given": dict(diar="sent_given", people=False),
    "text_only_participants": dict(diar=None, people=True),
    "diarised_estimated_participants": dict(diar="sent_est", people=True),
    "diarised_speakers_given_participants": dict(diar="sent_given", people=True),
    "oracle_speaker_names": dict(diar="truth", people=True),  # ceiling: the true speaker of every sentence, named
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    chosen = json.loads((ROOT / "results" / "diarize_chosen.json").read_text(encoding="utf8"))
    seg_chosen = json.loads((ROOT / "results" / "diarize_segmentation_chosen.json").read_text(encoding="utf8"))
    tr = FasterWhisperTranscriber(args.whisper, threads=args.threads)
    meetings = extract_eval.load_all(ROOT / "data" / "eval" / "extract")
    cache: dict[str, dict] = {}
    for m in meetings:
        wav = SYNTH / f"{m.name}.wav"
        samples = audio.decode(wav)
        wp = SYNTH / f"{m.name}.words.{args.whisper}.json"
        if not wp.exists():
            wp.write_text(json.dumps([w.model_dump() for w in tr.transcribe(samples)]), encoding="utf8")
        truth = json.loads((SYNTH / f"{m.name}.timing.json").read_text(encoding="utf8"))
        speakers = list(dict.fromkeys(s["speaker"] for s in truth["sentences"]))
        spans = [(w.start, w.end) for w in split_sentences(clean_words([Word(**w) for w in json.loads(wp.read_text(encoding="utf8"))]))]
        turns = {}
        for mode, k in (("est", None), ("given", len(speakers))):
            tp = SYNTH / f"{m.name}.turns.sent_{mode}.json"  # sentence-level method
            if not tp.exists():
                tp.write_text(json.dumps(diarize_sentences(samples, spans, ROOT / "models", num_speakers=k, threads=args.threads, **chosen["params"])), encoding="utf8")
            turns[f"sent_{mode}"] = [tuple(t) for t in json.loads(tp.read_text(encoding="utf8"))]
            tp = SYNTH / f"{m.name}.turns.{mode}.json"  # segmentation pipeline baseline
            if not tp.exists():
                tp.write_text(json.dumps(diarize(samples, ROOT / "models", num_speakers=-1 if k is None else k, threads=args.threads, **seg_chosen["params"])), encoding="utf8")
            turns[mode] = [tuple(t) for t in json.loads(tp.read_text(encoding="utf8"))]
        turns["truth"] = [(t["start"], t["end"], t["speaker"]) for t in truth["sentences"]]
        cache[m.name] = {"words": [Word(**w) for w in json.loads(wp.read_text(encoding="utf8"))], "dur": audio.duration(samples),
                         "turns": turns, "speakers": speakers, "meeting": m}  # fmt: skip
        print("ready", m.name, flush=True)

    # diarisation quality per split: DER, frame accuracy and speaker-count accuracy, speakers estimated vs given
    report = {"tuned_on": "dev", "settings": chosen["params"], "baseline_settings": seg_chosen["params"], "splits": {}}
    for split, spec in SPLITS.items():
        names = sorted(m.name for m in extract_eval.load_all(ROOT / "data" / "eval" / "extract", spec))
        report["splits"][split] = {}
        for mode in ("sent_est", "sent_given", "est", "given"):
            per = []
            for n in names:
                truth = json.loads((SYNTH / f"{n}.timing.json").read_text(encoding="utf8"))["sentences"]
                ref = [(t["start"], t["end"], t["speaker"]) for t in truth]
                r = der(ref, cache[n]["turns"][mode])
                r["speech"] = sum(e - st for st, e, _ in ref)
                per.append(r)
            report["splits"][split][{"sent_est": "sentence_estimated", "sent_given": "sentence_given", "est": "segmentation_estimated", "given": "segmentation_given"}[mode]] = summarise(per)
    (ROOT / "results" / "diarize_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")

    results = {"asr": args.whisper, "diarisation": chosen, "splits": {}}
    for split, spec in SPLITS.items():
        names = {m.name for m in extract_eval.load_all(ROOT / "data" / "eval" / "extract", spec)}
        results["splits"][split] = {"meetings": len(names), "variants": {}}
        for vname, cfg in VARIANTS.items():
            agg = {"correct": 0, "wrong": 0, "unassigned": 0, "tp": 0, "predicted": 0, "gold": 0, "decisions_tp": 0, "decisions_pred": 0, "decisions_gold": 0}
            for name in sorted(names):
                c = cache[name]
                meeting = build_meeting(
                    c["words"], c["dur"], c["meeting"].date, RulesExtractor(),
                    turns=c["turns"][cfg["diar"]] if cfg["diar"] else None,
                    people=c["speakers"] if cfg["people"] else None,
                )  # fmt: skip
                ev = demo_eval(meeting, ROOT / "data" / "eval" / "extract" / f"{name}.txt", SYNTH / f"{name}.timing.json", RulesExtractor())
                for k in ("correct", "wrong", "unassigned"):
                    agg[k] += ev["owners"][k]
                a, d = ev["action_items"], ev["decisions"]
                agg["tp"] += a["tp"]; agg["predicted"] += a["predicted"]; agg["gold"] += a["gold"]  # noqa: E702
                agg["decisions_tp"] += d["tp"]; agg["decisions_pred"] += d["predicted"]; agg["decisions_gold"] += d["gold"]  # noqa: E702
            p = agg["tp"] / agg["predicted"] if agg["predicted"] else 0.0
            r = agg["tp"] / agg["gold"] if agg["gold"] else 0.0
            agg.update(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)
            results["splits"][split]["variants"][vname] = agg
            print(f"{split:<8}{vname:<40}owners ok/wrong/none {agg['correct']}/{agg['wrong']}/{agg['unassigned']}  action F1 {agg['f1']:.2f}", flush=True)
    (ROOT / "results" / "audio_owners.json").write_text(json.dumps(results, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
