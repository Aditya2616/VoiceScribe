"""Command line interface: run, serve, fetch-models, synth-demo, build-demo, eval."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def cmd_run(a: argparse.Namespace) -> int:
    from .asr import FasterWhisperTranscriber
    from .export import export
    from .extract.base import get_extractor
    from .names import parse_people
    from .pipeline import process_file

    meeting, _ = process_file(
        a.audio,
        a.date or dt.date.today(),
        transcriber=FasterWhisperTranscriber(a.model),
        extractor=get_extractor(a.extractor),
        diarise=a.diarize,
        models_dir=Path(a.models_dir),
        people=parse_people(a.participants),
        progress=lambda stage, f: print(f"\r{stage:<10} {f * 100:5.1f}%", end="", file=sys.stderr, flush=True),
    )
    print(file=sys.stderr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    for fmt in ("json", "md", "srt", "vtt"):
        (out / f"{stem}.{fmt}").write_text(export(meeting, fmt)[0], encoding="utf8")
    print(f"{len(meeting.transcript.sentences)} sentences, {len(meeting.action_items)} action items, "
          f"{len(meeting.decisions)} decisions -> {out}/{stem}.*")  # fmt: skip
    for item in meeting.action_items:
        due = f" due {item.deadline_date}" if item.deadline_date else ""
        print(f"  [{item.source.start:6.1f}s] {item.owner or 'unassigned'}: {item.task}{due}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("voicescribe.server:app", host=a.host, port=a.port, log_level="info")
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    models = Path(a.models_dir)
    if a.whisper:
        from faster_whisper import WhisperModel

        for name in a.whisper:
            print(f"downloading whisper {name} ...")
            WhisperModel(name, device="cpu", compute_type="int8")
    if a.diarize:
        from .diarize import download

        print("downloading diarisation models ...")
        download(models)
    if a.local:
        from huggingface_hub import snapshot_download

        print(f"downloading {a.local} ...")
        snapshot_download(a.local)
    if a.synth:
        import urllib.request

        models.mkdir(parents=True, exist_ok=True)
        base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
        for f in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"):
            if not (models / f).exists():
                print(f"downloading {f} ...")
                urllib.request.urlretrieve(f"{base}/{f}", models / f)
    return 0


def cmd_synth(a: argparse.Namespace) -> int:
    from . import audio
    from .synth import synthesize

    out = Path(a.out)
    models = Path(a.models_dir)
    info = synthesize(Path(a.script), models / "kokoro-v1.0.int8.onnx", models / "voices-v1.0.bin", out)
    import soundfile as sf

    data, sr = sf.read(out / "meeting.wav", dtype="float32")
    audio.encode_mp3(data, out / "meeting.mp3", in_rate=sr)
    print(f"{info['duration']} s, {len(info['sentences'])} sentences -> {out}/meeting.mp3, meeting.timing.json")
    return 0


def cmd_build_demo(a: argparse.Namespace) -> int:
    """Run the real pipeline on the bundled synthetic meeting and write the recorded run for the web UI."""
    from . import audio
    from .asr import FasterWhisperTranscriber
    from .eval.extract_eval import load_meeting
    from .extract.base import get_extractor
    from .pipeline import process_file

    demo = Path(a.demo_dir)
    gold = load_meeting(demo / "script.txt")
    meeting, samples = process_file(
        demo / "meeting.mp3", gold.date, transcriber=FasterWhisperTranscriber(a.model),
        extractor=get_extractor(a.extractor), diarise=a.diarize, models_dir=Path(a.models_dir),
        title=gold.title, synthetic=True,
    )  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meeting.json").write_text(meeting.model_dump_json(), encoding="utf8")
    (out / "peaks.json").write_text(json.dumps({"per_second": 50, "peaks": audio.peaks(samples)}), encoding="utf8")
    shutil.copyfile(demo / "meeting.mp3", out / "meeting.mp3")
    print(f"recorded run: {meeting.transcript.model}, {len(meeting.action_items)} action items -> {out}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from .eval import extract_eval, speech_eval
    from .extract.base import get_extractor

    data = Path(a.data_dir)
    if getattr(a, "device", None):
        import os

        os.environ["VOICESCRIBE_DEVICE"] = a.device
    if a.what == "extract":
        ms = extract_eval.load_all(data / "eval" / "extract", a.split, with_speakers=a.speakers)
        ex = get_extractor(a.extractor)
        r = extract_eval.score(ms, ex, people_from_speakers=a.speakers)
        gen = getattr(ex, "_choose", None)
        if hasattr(gen, "device"):  # local model: record where and how it ran
            r["model"], r["device"], r["dtype"] = gen.model_name, gen.device, gen.dtype
        r["extractor"], r["split"], r["speakers_given"] = a.extractor, a.split or "all", a.speakers
    else:
        from .asr import FasterWhisperTranscriber

        tr = FasterWhisperTranscriber(a.model)
        if a.what == "wer":
            r = speech_eval.librispeech_wer(data / "eval" / "wer", tr)
        else:
            r, _ = speech_eval.run_demo_eval(
                data / "demo" / "meeting.mp3", data / "demo" / "script.txt", data / "demo" / "meeting.timing.json",
                tr, get_extractor(a.extractor), diarise=a.diarize,
            )  # fmt: skip
    text = json.dumps(r, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="voicescribe", description="Meetings in, clean action items out.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="transcribe an audio file and extract action items")
    r.add_argument("audio")
    r.add_argument("--date", type=_date, help="meeting date, YYYY-MM-DD (default: today)")
    r.add_argument("--model", default=None, help="tiny.en | base.en | small.en (default: VOICESCRIBE_WHISPER or base.en)")
    r.add_argument("--extractor", default=None, help="rules | local | local-verify | openai | anthropic")
    r.add_argument("--diarize", action="store_true")
    r.add_argument("--participants", default=None, help="comma-separated first names: the closed set for owners")
    r.add_argument("--models-dir", default="models")
    r.add_argument("--out", default="outputs")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8107)
    s.set_defaults(fn=cmd_serve)

    f = sub.add_parser("fetch-models", help="download models")
    f.add_argument("--whisper", nargs="*", default=["base.en"])
    f.add_argument("--diarize", action="store_true")
    f.add_argument("--local", nargs="?", const="Qwen/Qwen2.5-0.5B-Instruct", default=None)
    f.add_argument("--synth", action="store_true", help="Kokoro-82M, only needed to regenerate the demo audio")
    f.add_argument("--models-dir", default="models")
    f.set_defaults(fn=cmd_fetch)

    y = sub.add_parser("synth-demo", help="regenerate the synthetic demo meeting audio")
    y.add_argument("--script", default="data/demo/script.txt")
    y.add_argument("--out", default="outputs")
    y.add_argument("--models-dir", default="models")
    y.set_defaults(fn=cmd_synth)

    b = sub.add_parser("build-demo", help="record a real run on the demo meeting for the web UI")
    b.add_argument("--model", default="small.en")
    b.add_argument("--extractor", default="rules")
    b.add_argument("--diarize", action="store_true")
    b.add_argument("--demo-dir", default="data/demo")
    b.add_argument("--out", default="web/demo")
    b.add_argument("--models-dir", default="models")
    b.set_defaults(fn=cmd_build_demo)

    e = sub.add_parser("eval", help="reproduce the numbers in the README")
    e.add_argument("what", choices=["wer", "extract", "demo"])
    e.add_argument("--model", default="small.en")
    e.add_argument("--extractor", default="rules")
    e.add_argument("--device", default=None, help="local extractor: cuda | cpu (default: automatic)")
    e.add_argument("--split", default=None, help="extract only: dev | test | test2 (default: all)")
    e.add_argument("--speakers", action="store_true", help="extract only: give the extractor speaker names")
    e.add_argument("--diarize", action="store_true", help="demo only")
    e.add_argument("--data-dir", default="data")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
