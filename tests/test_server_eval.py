import datetime as dt
import time

import pytest
from conftest import ROOT, words_from_text
from fastapi.testclient import TestClient

from voicescribe.asr import StubTranscriber
from voicescribe.audio import AudioError, decode, peaks
from voicescribe.eval.extract_eval import load_all, load_meeting, score
from voicescribe.eval.wer import corpus_wer, normalize, wer
from voicescribe.extract.base import RulesExtractor
from voicescribe.server import create_app

SPOKEN = "Okay let us start. Rohan, can you ship the invite flow by next Friday? Sure, I will do that."


def client_with_stub():
    words = words_from_text(SPOKEN)
    return TestClient(create_app(StubTranscriber(words)))


def wait_done(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health():
    r = client_with_stub().get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["max_upload_mb"] == 50
    assert set(body["hosted_keys"]) == {"openai", "anthropic"}


def test_upload_process_and_export(tiny_wav):
    c = client_with_stub()
    with tiny_wav.open("rb") as fh:
        r = c.post("/api/jobs", files={"file": ("standup.wav", fh, "audio/wav")}, data={"date": "2026-11-12"})
    assert r.status_code == 202
    job = wait_done(c, r.json()["id"])
    assert job["state"] == "done", job
    meeting = job["meeting"]
    assert meeting["title"] == "standup" and len(job["peaks"]) == 100  # 2 s at 50 per second
    (item,) = meeting["action_items"]
    assert item["owner"] == "Rohan" and item["deadline_date"] == "2026-11-20"
    md = c.get(f"/api/jobs/{r.json()['id']}/export", params={"format": "md"})
    assert md.status_code == 200 and "Rohan" in md.text
    srt = c.get(f"/api/jobs/{r.json()['id']}/export", params={"format": "srt"})
    assert "-->" in srt.text
    assert c.get(f"/api/jobs/{r.json()['id']}/export", params={"format": "docx"}).status_code == 422


def test_bad_inputs_are_rejected_cleanly(tiny_wav, monkeypatch):
    c = client_with_stub()
    assert c.get("/api/jobs/nope").status_code == 404
    r = c.post("/api/jobs", files={"file": ("x.wav", b"", "audio/wav")})
    assert r.status_code == 422
    r = c.post("/api/jobs", files={"file": ("x.wav", tiny_wav.read_bytes(), "audio/wav")}, data={"date": "not-a-date"})
    assert r.status_code == 422
    r = c.post("/api/jobs", files={"file": ("x.wav", tiny_wav.read_bytes(), "audio/wav")}, data={"extractor": "bogus"})
    assert r.status_code == 422
    monkeypatch.setattr("voicescribe.server.MAX_UPLOAD_BYTES", 1000)
    r = c.post("/api/jobs", files={"file": ("x.wav", tiny_wav.read_bytes(), "audio/wav")})
    assert r.status_code == 413
    r = c.post("/api/jobs", files={"file": ("x.wav", b"this is not audio at all" * 50, "audio/wav")})
    # larger than the patched cap, so it is refused before decoding
    assert r.status_code == 413


def test_undecodable_audio_becomes_a_job_error_and_temp_file_is_removed(monkeypatch):
    import glob
    import tempfile

    c = client_with_stub()
    r = c.post("/api/jobs", files={"file": ("x.wav", b"this is not audio at all" * 50, "audio/wav")})
    job = wait_done(c, r.json()["id"])
    assert job["state"] == "error" and "decode" in job["error"]
    assert not glob.glob(f"{tempfile.gettempdir()}/voicescribe-*/upload.bin")


def test_audio_decode_and_peaks(tiny_wav):
    samples = decode(tiny_wav)
    assert samples.dtype.name == "float32" and abs(len(samples) - 32000) < 50
    p = peaks(samples)
    assert len(p) == 100 and max(p) <= 255 and min(p) > 0
    with pytest.raises(AudioError):
        decode(ROOT / "README-missing.wav")


# ---- evaluation code -------------------------------------------------------------------------------


def test_wer_basics_and_number_normalisation():
    assert wer("the cat sat", "the cat sat") == 0
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert wer("a b c d", "a c d") == pytest.approx(0.25)  # one deletion
    assert wer("about two hundred", "About 200.") == 0
    assert normalize("twenty fifth of May") == ["25th", "of", "may"]
    assert corpus_wer([("a b", "a b"), ("c d", "c x")]) == pytest.approx(0.25)


def test_wer_matches_jiwer_when_available():
    jiwer = pytest.importorskip("jiwer")
    ref, hyp = "she sells sea shells by the sea shore", "she sell sea shell on the shore"
    assert wer(ref, hyp) == pytest.approx(jiwer.wer(ref, hyp))


def test_labelled_meetings_load_and_scoring_counts():
    meetings = load_all(ROOT / "data" / "eval" / "extract")
    assert {m.split for m in meetings} == {"dev", "test", "test2"}
    m = load_meeting(ROOT / "data" / "eval" / "extract" / "dev1.txt", with_speakers=True)
    assert m.date == dt.date(2026, 3, 4) and m.transcript.sentences[3].speaker == "Ben"
    assert [g.kind for g in m.gold].count("action") == 3
    report = score([m], RulesExtractor(), people_from_speakers=True)
    assert report["action_items"]["gold"] == 3 and report["action_items"]["tp"] == 3
    assert report["owner_accuracy"] == 1.0 and report["due_date_accuracy"] == 1.0


def test_scoring_penalises_false_positives_and_misses():
    class Silent:
        method = "silent"

        def extract(self, t, d, people=None):
            return [], [], None

    m = load_meeting(ROOT / "data" / "eval" / "extract" / "dev1.txt")
    r = score([m], Silent())
    assert r["action_items"]["recall"] == 0 and r["decisions"]["recall"] == 0 and r["action_items"]["predicted"] == 0
