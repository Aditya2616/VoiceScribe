"""Keep README.md provably consistent with the results files, the recorded demo and the code.

    python scripts/check_numbers.py --check    # exit 1 if anything is out of sync (this is what the test runs)
    python scripts/check_numbers.py --write    # regenerate the generated parts of README.md

Two kinds of generated text live in README.md:

* blocks:  <!--b:name--> ... <!--/b:name-->   (the result tables)
* inline:  <!--n:key-->value<!--/n-->          (every number quoted in prose)

Both are computed here from results/*.json, docs/demo/meeting.json, data/eval and the code (limits, defaults), never
typed by hand. --check also verifies that the commands, flags, paths and environment variables the README mentions
exist, that docs/ is an up-to-date copy of web/, and that the recorded demo agrees with its results file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

README = ROOT / "README.md"
RES = ROOT / "results"
FUTURE_OUTPUTS = {"results/speed_cpu.json"}  # written by scripts/rerecord_demo.py once the machine is quiet
SPLITS = {"dev": "dev", "test": "test", "test2": "test2", "held": "held", "heldout": "test2,held"}


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf8"))


def ext(method: str, split: str) -> dict:
    return load(f"extract_{method}_{split}.json")


def f2(x: float) -> str:
    return f"{x:.2f}"


def prf(r: dict) -> str:
    a = r["action_items"]
    return f"{f2(a['precision'])} / {f2(a['recall'])} / {f2(a['f1'])}"


# ---------------------------------------------------------------------------------------------- values


def compute() -> dict[str, str]:
    from voicescribe import audio as audio_mod
    from voicescribe import server
    from voicescribe.eval import extract_eval
    from voicescribe.eval.speech_eval import evaluate_recorded
    from voicescribe.extract import local

    v: dict[str, str] = {}
    # --- LibriSpeech WER
    refs = [ln.split("\t")[0] for ln in (ROOT / "data/eval/wer/refs.tsv").read_text(encoding="utf8").splitlines() if ln]
    small, base = load("wer_small.en.json"), load("wer_base.en.json")
    assert small["utterances"] == base["utterances"] == len(refs)
    v["libri_utts"] = str(len(refs))
    v["libri_speakers"] = str(len({r.split("-")[0] for r in refs}))
    v["libri_words"] = str(small["reference_words"])
    v["wer_small"] = f"{100 * small['wer']:.2f}%"
    v["wer_base"] = f"{100 * base['wer']:.2f}%"
    assert small["reference_words"] == base["reference_words"]

    # --- recorded demo run vs its script
    demo = load("demo_recorded.json")
    live = evaluate_recorded(ROOT / "docs/demo/meeting.json", ROOT / "data/demo/script.txt", ROOT / "data/demo/meeting.timing.json")
    for k in live:
        assert demo[k] == live[k], f"results/demo_recorded.json is stale for {k}: re-run scripts/rerecord_demo.py"
    raw = demo["raw_asr"]
    v["demo_words"] = str(raw["reference_words"])
    v["demo_raw_wer"] = f"{100 * raw['wer']:.2f}%"
    v["demo_raw_subs"] = str(raw["substitutions"])
    v["demo_named_wer"] = f"{100 * demo['wer']:.2f}%"
    v["demo_named_subs"] = str(demo["substitutions"])
    b = raw["boundary_error_ms"]
    v["ts_n"] = str(b["n"])
    v["ts_median"] = f"{b['median_abs']:.0f} ms"
    v["ts_p90"] = f"{b['p90_abs']:.0f} ms"
    v["ts_within"] = f"{100 * b['within_250ms']:.0f}%"
    v["ts_end"] = f"{abs(b['mean_end_signed']):.0f} ms {'early' if b['mean_end_signed'] < 0 else 'late'}"
    v["ts_start"] = f"{abs(b['mean_start_signed']):.0f} ms {'early' if b['mean_start_signed'] < 0 else 'late'}"
    a, d = demo["action_items"], demo["decisions"]
    v["demo_items"] = f"{a['tp']} of {a['gold']}"
    v["demo_items_n"] = str(a["predicted"])
    v["demo_decisions"] = f"{d['tp']} of {d['gold']}"
    v["demo_owners"] = f"{round(demo['owner_accuracy'] * a['tp'])} of {a['tp']}"
    v["demo_dates"] = f"{round(demo['due_date_accuracy'] * a['tp'])} of {a['tp']}"
    v["demo_model"] = demo["asr_model"].split("/")[-1]
    v["demo_participants"] = ", ".join(json.loads((ROOT / "docs/demo/meeting.json").read_text(encoding="utf8"))["participants"])

    # --- labelled meetings
    data = ROOT / "data/eval/extract"
    stats = {}
    for name, spec in SPLITS.items():
        ms = extract_eval.load_all(data, spec)
        stats[name] = (len(ms), sum(len(m.transcript.sentences) for m in ms),
                       sum(g.kind == "action" for m in ms for g in m.gold), sum(g.kind == "decision" for m in ms for g in m.gold))  # fmt: skip
    for name, (m, s, a_, d_) in stats.items():
        v[f"n_{name}"], v[f"sent_{name}"], v[f"act_{name}"], v[f"dec_{name}"] = str(m), str(s), str(a_), str(d_)
    total_m = sum(stats[k][0] for k in ("dev", "test", "test2", "held"))
    total_s = sum(stats[k][1] for k in ("dev", "test", "test2", "held"))
    v["n_all"] = str(total_m)
    v["mean_sent"] = str(round(total_s / total_m))

    # --- heldout comparisons used in prose
    rules, o05, o15 = ext("rules", "heldout"), ext("owners_0.5B", "heldout"), ext("owners_1.5B", "heldout")
    v1, v05 = ext("verify_1.5B", "heldout"), ext("verify_0.5B", "heldout")
    v["rules_prf"] = prf(rules)
    v["rules_p"], v["rules_r"] = f2(rules["action_items"]["precision"]), f2(rules["action_items"]["recall"])
    ro = rules["owners"]
    v["rules_owner_ok"], v["rules_owner_n"], v["rules_owner_wrong"], v["rules_owner_none"] = (
        str(ro["correct"]), str(ro["n"]), str(ro["wrong"]), str(ro["unassigned"]))  # fmt: skip
    v["o05_resolved"] = str(ro["unassigned"] - o05["owners"]["unassigned"])
    v["o15_resolved"] = str(ro["unassigned"] - o15["owners"]["unassigned"])
    v["o15_wrong"] = str(o15["owners"]["wrong"] - ro["wrong"])
    v["v15_prf"], v["v05_p"] = prf(v1), f2(v05["action_items"]["precision"])
    v["v05_prf"] = prf(v05)

    # --- diarisation
    dz = load("diarize_demo.json")
    u, g = dz["speakers_unknown"], dz["speakers_given_4"]
    v["diar_acc"] = f"{100 * u['accuracy']:.0f}%"
    v["diar_found"], v["diar_true"] = str(u["predicted_speakers"]), str(u["true_speakers"])
    v["diar_secs"] = str(int(round((u["seconds"] + g["seconds"]) / 2, -1)))
    same = abs(u["accuracy"] - g["accuracy"]) < 0.005 and u["predicted_speakers"] == g["predicted_speakers"]
    v["diar_given"] = "the same" if same else f"{100 * g['accuracy']:.0f}% with {g['predicted_speakers']} speakers"

    # --- limits and defaults from the code
    v["max_mb"] = str(server.MAX_UPLOAD_BYTES // (1024 * 1024))
    v["max_min"] = str(audio_mod.MAX_DURATION_S // 60)
    import inspect

    w = inspect.signature(local._window).parameters
    v["win_before"], v["win_after"] = str(w["before"].default), str(w["after"].default)
    v["owner_thr"] = str(inspect.signature(local.LocalExtractor.__init__).parameters["owner_threshold"].default)
    rd = __import__("rerecord_demo")
    v["quiet_pct"], v["threads"] = f"{100 * rd.QUIET_BELOW:.0f}%", str(rd.DEFAULT_THREADS)
    v["port"] = re.search(r'"--port", type=int, default=(\d+)', (ROOT / "src/voicescribe/cli.py").read_text(encoding="utf8")).group(1)
    asr_src = (ROOT / "src/voicescribe/asr.py").read_text(encoding="utf8")
    v["default_whisper"] = re.search(r'"VOICESCRIBE_WHISPER", "([\w.]+)"', asr_src).group(1)
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf8")
    v["av_pin"] = re.search(r'"av(>=\d+,<\d+)"', pyproject).group(1)
    cache = sorted((Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-base.en").glob("snapshots/*/model.bin"))
    v["base_mb"] = f"{round(cache[0].stat().st_size / 1e6)} MB" if cache else "145 MB"
    return v


# ---------------------------------------------------------------------------------------------- tables

LABELS = [
    ("rules (default)", "rules"),
    ("rules + 0.5B owner resolution", "owners_0.5B"),
    ("rules + 1.5B owner resolution", "owners_1.5B"),
    ("rules + 0.5B verifier", "verify_0.5B"),
    ("rules + 1.5B verifier", "verify_1.5B"),
]
SET_NAMES = [("dev", "dev"), ("heldout", "**held-out**"), ("held", "`held` only")]


def extract_table() -> str:
    lines = ["| Extractor | Set (meetings) | Action P / R / F1 | Decision F1 | Owners ok / wrong / none | Device |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for label, method in LABELS:
        for split, name in SET_NAMES:
            r = ext(method, split)
            o = r["owners"]
            dev = f"{r['device']} {r['dtype']}" if "device" in r else "cpu (no model)"
            lines.append(f"| {label} | {name} ({r['meetings']}) | {prf(r)} | {f2(r['decisions']['f1'])} | "
                         f"{o['correct']} / {o['wrong']} / {o['unassigned']} | {dev} |")  # fmt: skip
    return "\n".join(lines)


def speech_table(v: dict[str, str]) -> str:
    return "\n".join([
        "| Set | Model | Words | WER |",
        "|---|---|---|---|",
        f"| LibriSpeech test-clean subset ({v['libri_utts']} utterances, read speech) | small.en | {v['libri_words']} | {v['wer_small']} |",
        f"| same | base.en | {v['libri_words']} | {v['wer_base']} |",
        f"| Synthetic demo meeting, recogniser output as is (scored against its script) | {v['demo_model']} | {v['demo_words']} | {v['demo_raw_wer']} ({v['demo_raw_subs']} substitution{'' if v['demo_raw_subs'] == '1' else 's'}) |",
        f"| Synthetic demo meeting, recorded run (names corrected from the attendee list) | {v['demo_model']} | {v['demo_words']} | {v['demo_named_wer']} ({v['demo_named_subs']} substitution{'' if v['demo_named_subs'] == '1' else 's'}) |",
    ])  # fmt: skip


# ---------------------------------------------------------------------------------------------- apply

INLINE = re.compile(r"<!--n:(\w+)-->(.*?)<!--/n-->", re.S)
BLOCK = re.compile(r"(<!--b:(\w+)-->\n)(.*?)(\n<!--/b:\2-->)", re.S)


def render(text: str, v: dict[str, str]) -> str:
    blocks = {"speech_table": speech_table(v), "extract_table": extract_table()}

    def inline(m: re.Match) -> str:
        key = m.group(1)
        if key not in v:
            raise KeyError(f"README uses unknown number key {key!r}")
        return f"<!--n:{key}-->{v[key]}<!--/n-->"

    def block(m: re.Match) -> str:
        return f"{m.group(1)}{blocks[m.group(2)]}{m.group(4)}"

    return INLINE.sub(inline, BLOCK.sub(block, text))


# ---------------------------------------------------------------------------------------------- checks


def check_references(text: str) -> list[str]:
    """Commands, flags, paths and environment variables named in the README must exist."""
    from voicescribe.cli import build_parser

    problems: list[str] = []
    parser = build_parser()
    subs = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    code = re.findall(r"`([^`\n]+)`", text) + re.findall(r"^\s*(voicescribe [^#\n]+)", text, re.M)
    for snippet in code:
        m = re.match(r"(?:\w+=\S+ )*voicescribe (\S+)(.*)", snippet.strip())
        if m:
            sub = m.group(1)
            if sub not in subs:
                problems.append(f"README runs `voicescribe {sub}`, which is not a command")
                continue
            known = {o for a in subs[sub]._actions for o in a.option_strings}
            for flag in re.findall(r"(--[\w-]+)", m.group(2)):
                if flag not in known:
                    problems.append(f"README uses `voicescribe {sub} {flag}`, which does not exist")
    for flag_line in re.findall(r"`(--[\w-]+)`", text):
        if not any(flag_line in a.option_strings for sp in subs.values() for a in sp._actions):
            problems.append(f"README mentions option {flag_line}, which no command has")
    for script in set(re.findall(r"python (scripts/[\w./]+\.py)", text)):
        if not (ROOT / script).exists():
            problems.append(f"README runs {script}, which does not exist")
    for path in set(re.findall(r"`((?:docs|results|data|scripts|src|web|tests)/[^`\s<>*]*)`", text)):
        if "<" not in path and path not in FUTURE_OUTPUTS and not (ROOT / path).exists():
            problems.append(f"README names {path}, which does not exist")
    for png in re.findall(r"docs/assets/[\w-]+\.png", text):
        if not (ROOT / png).exists():
            problems.append(f"README shows {png}, which does not exist")
    sources = "\n".join(p.read_text(encoding="utf8", errors="ignore") for p in [*(ROOT / "src").rglob("*.py"), ROOT / ".env.example"])
    for var in set(re.findall(r"`([A-Z][A-Z0-9]*_[A-Z0-9_]+)`", text)):
        if var not in sources:
            problems.append(f"README mentions {var}, which the code does not read")
    return problems


def check_static_facts(text: str, v: dict[str, str]) -> list[str]:
    """Values quoted inside code blocks (where inline markers would show) must equal the code's."""
    need = {
        f"http://127.0.0.1:{v['port']}": "the quickstart URL and the server's default port",
        f"(about {v['base_mb']})": "the base.en download size",
        f"--whisper {v['default_whisper']}": "the default Whisper size in the quickstart",
    }
    problems = [f"README should contain {k!r} ({why})" for k, why in need.items() if k not in text]
    if f"http://127.0.0.1:{v['port']}" not in (ROOT / "web/app.js").read_text(encoding="utf8"):
        problems.append("the UI's local-server hint names a different port from the CLI default")
    return problems


def check_docs_sync() -> list[str]:
    problems = []
    for src in (ROOT / "web").rglob("*"):
        if not src.is_file():
            continue
        dst = ROOT / "docs" / src.relative_to(ROOT / "web")
        if not dst.exists():
            problems.append(f"docs/{src.relative_to(ROOT / 'web')} is missing: run scripts/build_docs.py")
        elif src.name == "index.html":
            from build_docs import STATIC_META

            want = src.read_bytes().replace(b"<head>", STATIC_META.encode(), 1)
            if dst.read_bytes().replace(b"\r\n", b"\n") != want.replace(b"\r\n", b"\n"):
                problems.append("docs/index.html differs from web/index.html: run scripts/build_docs.py")
        elif dst.read_bytes().replace(b"\r\n", b"\n") != src.read_bytes().replace(b"\r\n", b"\n"):
            problems.append(f"docs/{dst.relative_to(ROOT / 'docs')} differs from web/: run scripts/build_docs.py")
    return problems


def check_demo_and_results() -> list[str]:
    from voicescribe.eval.extract_eval import script_participants
    from voicescribe.schema import Meeting

    problems = []
    m = Meeting.model_validate_json((ROOT / "docs/demo/meeting.json").read_text(encoding="utf8"))
    if m.participants != script_participants(ROOT / "data/demo/script.txt"):
        problems.append("the recorded demo was not made with the script's attendee list")
    if not m.synthetic:
        problems.append("the recorded demo is not labelled synthetic")
    for p in RES.glob("extract_*_*.json"):
        r = json.loads(p.read_text(encoding="utf8"))
        if re.match(r"extract_(owners|verify)_", p.name) and not {"device", "dtype", "model"} <= r.keys():
            problems.append(f"{p.name} does not record model, device and dtype")
    ui = (ROOT / "web/index.html").read_text(encoding="utf8") + (ROOT / "web/app.js").read_text(encoding="utf8")
    for needle in ("Synthetic speech", "Kokoro-82M", "extractive", "recorded run", "participants given"):
        if needle not in ui:
            problems.append(f"the UI no longer says {needle!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    text = README.read_text(encoding="utf8").replace("\r\n", "\n")
    want = render(text, compute())
    if args.write:
        README.write_text(want, encoding="utf8", newline="\n")
        print("README.md regenerated")
        return 0
    problems = []
    if want != text:
        import difflib

        diff = [ln for ln in difflib.unified_diff(text.splitlines(), want.splitlines(), "README.md", "generated", lineterm="", n=0)
                if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]  # fmt: skip
        problems.append("README.md is out of sync with the results; run scripts/check_numbers.py --write:\n  " + "\n  ".join(diff[:12]))
    problems += check_references(text) + check_static_facts(text, compute()) + check_docs_sync() + check_demo_and_results()
    for p in problems:
        print("PROBLEM:", p)
    print("all numbers and references check out" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
