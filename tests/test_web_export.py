"""The browser exports (web/export.js) must match the Python exports on the bundled recorded run."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from voicescribe.export import to_markdown, to_srt, to_vtt
from voicescribe.schema import Meeting

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "web" / "demo" / "meeting.json"

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not DEMO.exists(), reason="needs node and the demo run")

SCRIPT = """
import { toMarkdown, toSrt, toVtt } from %s;
import fs from "node:fs";
const m = JSON.parse(fs.readFileSync(%s, "utf8"));
process.stdout.write(JSON.stringify({ md: toMarkdown(m), srt: toSrt(m), vtt: toVtt(m) }));
"""


def test_js_exports_match_python():
    code = SCRIPT % (json.dumps((ROOT / "web" / "export.js").as_uri()), json.dumps(str(DEMO)))
    out = subprocess.run(
        ["node", "--input-type=module", "-e", code], capture_output=True, text=True, check=True, timeout=60
    )
    js = json.loads(out.stdout)
    meeting = Meeting.model_validate_json(DEMO.read_text(encoding="utf8"))
    assert js["srt"] == to_srt(meeting)
    assert js["vtt"] == to_vtt(meeting)
    assert js["md"] == to_markdown(meeting)
