"""Synthesise every labelled meeting (dev, test, test2, held) with Kokoro-82M, one distinct voice per speaker.

The audio is synthetic speech with exact speaker turns and sentence timings, written to outputs/synth/ (git-ignored;
regenerate with this script). It is the ground truth for diarisation and for owner resolution from audio.

    python scripts/synth_eval_meetings.py [--threads 2] [--only held1 held2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicescribe.eval.extract_eval import load_meeting  # noqa: E402
from voicescribe.synth import assign_voices, synthesize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "synth"))
    args = ap.parse_args()
    out = Path(args.out)
    for k, path in enumerate(sorted((ROOT / "data" / "eval" / "extract").glob("*.txt"))):
        if args.only and path.stem not in args.only:
            continue
        if (out / f"{path.stem}.timing.json").exists():
            print("have", path.stem)
            continue
        m = load_meeting(path, with_speakers=True)
        speakers = list(dict.fromkeys(s.speaker for s in m.transcript.sentences))
        info = synthesize(path, ROOT / "models" / "kokoro-v1.0.int8.onnx", ROOT / "models" / "voices-v1.0.bin", out,
                          name=path.stem, voice_map=assign_voices(speakers, offset=k), threads=args.threads)  # fmt: skip
        print(path.stem, info["duration"], "s", info["voices"], flush=True)


if __name__ == "__main__":
    main()
