"""Speaker diarisation accuracy on the synthetic demo meeting (true speaker turns are known from the TTS script).

Frame accuracy (10 ms) over frames with speech, after the best one-to-one mapping of predicted to true speakers.
Run with the number of speakers unknown (default clustering) and given (4)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicescribe import audio  # noqa: E402
from voicescribe.diarize import diarize  # noqa: E402
from voicescribe.eval.speech_eval import speaker_accuracy  # noqa: E402


def main() -> None:
    samples = audio.decode(ROOT / "data/demo/meeting.mp3")
    sents = json.loads((ROOT / "data/demo/meeting.timing.json").read_text())["sentences"]
    truth = [(s["start"], s["end"], s["speaker"]) for s in sents]
    out = {}
    for label, k in (("speakers_unknown", -1), ("speakers_given_4", 4)):
        t = time.perf_counter()
        turns = diarize(samples, ROOT / "models", num_speakers=k)
        r = speaker_accuracy(truth, turns)
        r["seconds"] = round(time.perf_counter() - t, 1)
        out[label] = r
        print(label, round(r["accuracy"], 3), r["predicted_speakers"], "speakers", r["seconds"], "s")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/diarize_demo.json").write_text(json.dumps(out, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
