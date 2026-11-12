"""One command to measure transcription speed and re-record the demo run.

    python scripts/rerecord_demo.py [--threads 3] [--models tiny.en base.en small.en]

For each Whisper size it transcribes the demo meeting on the CPU (int8, N threads), reports the real-time factor
(transcription seconds per audio second; below 1 is faster than real time), the WER against the script, and the
overall CPU busy fraction measured just before the run. The figure only counts as a clean measurement when the CPU
was mostly idle (busy fraction below 25%); otherwise it is recorded with a "loaded machine" label and the UI does
not show it. Then it records the demo run with small.en, copies the web UI into docs/ and writes
results/speed_cpu.json.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicescribe import audio  # noqa: E402
from voicescribe.asr import FasterWhisperTranscriber  # noqa: E402
from voicescribe.eval.diar_metrics import der  # noqa: E402
from voicescribe.eval.extract_eval import load_meeting, script_participants  # noqa: E402
from voicescribe.eval.speech_eval import asr_scores, demo_eval, evaluate_recorded  # noqa: E402
from voicescribe.extract.base import get_extractor  # noqa: E402
from voicescribe.pipeline import process_file  # noqa: E402

QUIET_BELOW = 0.25
DEFAULT_THREADS = 3


def cpu_busy(seconds: float = 3.0) -> float:
    """Fraction of all logical cores busy over `seconds` (Windows: GetSystemTimes; elsewhere: load average)."""
    if os.name == "nt":

        class FT(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

        def sample() -> tuple[int, int, int]:
            idle, kern, user = FT(), FT(), FT()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
            val = lambda f: (f.hi << 32) | f.lo  # noqa: E731
            return val(idle), val(kern), val(user)

        i0, k0, u0 = sample()
        time.sleep(seconds)
        i1, k1, u1 = sample()
        total = (k1 - k0) + (u1 - u0)  # kernel time includes idle time
        return round(1 - (i1 - i0) / total, 3) if total else 0.0
    return round(os.getloadavg()[0] / (os.cpu_count() or 1), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--models", nargs="*", default=["tiny.en", "base.en", "small.en"])
    ap.add_argument("--record-with", default="small.en")
    args = ap.parse_args()
    demo = ROOT / "data" / "demo"
    samples = audio.decode(demo / "meeting.mp3")
    dur = audio.duration(samples)
    rows = []
    for name in args.models:
        tr = FasterWhisperTranscriber(name, threads=args.threads)
        tr._load()  # model load is not part of the speed figure
        busy = cpu_busy()
        meeting, _ = process_file(demo / "meeting.mp3", load_meeting(demo / "script.txt").date, transcriber=tr,
                                  extractor=get_extractor("rules"), synthetic=True)  # fmt: skip
        asr_s = meeting.timings["asr_s"]
        ev = demo_eval(meeting, demo / "script.txt", demo / "meeting.timing.json", get_extractor("rules"))
        rows.append({
            "model": name, "audio_s": round(dur, 1), "asr_s": round(asr_s, 1), "rtf": round(asr_s / dur, 3),
            "wer_vs_script": round(ev["wer"], 4), "threads": args.threads, "cpu_busy_before": busy,
            "clean": busy < QUIET_BELOW,
        })  # fmt: skip
        print(rows[-1], flush=True)
    out = {"date": dt.date.today().isoformat(), "device": "cpu", "compute": "int8", "audio": "demo meeting (synthetic)",
           "rows": rows, "note": "rtf = transcription seconds per audio second; only rows with clean=true were measured on a mostly idle CPU"}  # fmt: skip
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "speed_cpu.json").write_text(json.dumps(out, indent=2), encoding="utf8")

    # record the demo run with the chosen size and mark whether its timing came from a quiet machine
    tr = FasterWhisperTranscriber(args.record_with, threads=args.threads)
    tr._load()
    busy = cpu_busy()
    gold = load_meeting(demo / "script.txt")
    raw: list = []
    meeting, samples = process_file(demo / "meeting.mp3", gold.date, transcriber=tr, extractor=get_extractor("rules"),
                                    title=gold.title, synthetic=True, on_words=raw.extend, diarise=True,
                                    models_dir=ROOT / "models",
                                    people=script_participants(demo / "script.txt"))  # fmt: skip
    meeting.timings["quiet_machine"] = 1.0 if busy < QUIET_BELOW else 0.0
    meeting.timings["cpu_busy_before"] = busy
    web = ROOT / "web" / "demo"
    (web / "meeting.json").write_text(meeting.model_dump_json(), encoding="utf8")
    (web / "peaks.json").write_text(json.dumps({"per_second": 50, "peaks": audio.peaks(samples)}), encoding="utf8")
    shutil.copyfile(demo / "meeting.mp3", web / "meeting.mp3")
    ev = evaluate_recorded(web / "meeting.json", demo / "script.txt", demo / "meeting.timing.json")
    truth = json.loads((demo / "meeting.timing.json").read_text(encoding="utf8"))["sentences"]
    raw_asr = asr_scores(raw, truth)  # the recogniser's own words, before the attendee list corrects names
    raw_asr.pop("hypothesis")
    ev["raw_asr"] = raw_asr
    ref = [(t["start"], t["end"], t["speaker"]) for t in truth]
    hyp = [(s.start, s.end, s.speaker) for s in meeting.transcript.sentences if s.speaker]
    d = der(ref, hyp)  # speaker identification of the recorded run: labels are names where bound, else "Speaker N"
    ev["diarisation"] = {k: d[k] for k in ("der", "frame_accuracy", "true_speakers", "predicted_speakers")}
    (ROOT / "results" / "demo_recorded.json").write_text(json.dumps(ev, indent=2), encoding="utf8")
    print(f"recorded {meeting.transcript.model}: rtf {meeting.timings['rtf']}, cpu busy {busy}, "
          f"quiet={bool(meeting.timings['quiet_machine'])}")  # fmt: skip
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_docs.py")], check=True)


if __name__ == "__main__":
    main()
