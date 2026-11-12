"""FastAPI service: upload or record audio, follow progress, fetch the structured meeting, export."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from . import audio as audio_mod
from .asr import FasterWhisperTranscriber, Transcriber
from .export import export
from .extract.base import get_extractor
from .pipeline import process_file
from .schema import Meeting

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_JOBS = 20
WEB_DIR = Path(os.environ.get("VOICESCRIBE_WEB", Path(__file__).resolve().parents[2] / "web"))
MODELS_DIR = Path(os.environ.get("VOICESCRIBE_MODELS", "models"))


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    whisper_model: str
    extractor: str
    diarization_available: bool
    hosted_keys: dict[str, bool]
    max_upload_mb: int
    max_minutes: int


class JobStatus(BaseModel):
    id: str
    state: Literal["queued", "running", "done", "error"]
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None


class JobResult(JobStatus):
    meeting: Meeting | None = None
    peaks: list[int] | None = None
    peaks_per_second: int = 50


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.state: str = "queued"
        self.stage = "queued"
        self.progress = 0.0
        self.error: str | None = None
        self.meeting: Meeting | None = None
        self.peaks: list[int] | None = None

    def status(self) -> JobStatus:
        return JobStatus(id=self.id, state=self.state, stage=self.stage, progress=self.progress, error=self.error)  # type: ignore[arg-type]


def create_app(transcriber: Transcriber | None = None) -> FastAPI:
    app = FastAPI(title="VoiceScribe", version=__version__)
    jobs: OrderedDict[str, Job] = OrderedDict()
    pool = ThreadPoolExecutor(max_workers=1)  # the CPU is the bottleneck: one job at a time
    lock = threading.Lock()
    holder: dict[str, Any] = {"transcriber": transcriber}

    def get_transcriber() -> Transcriber:
        if holder["transcriber"] is None:
            holder["transcriber"] = FasterWhisperTranscriber()
        return holder["transcriber"]

    def run_job(job: Job, path: Path, tmp: Path, date: dt.date, diarise: bool, extractor: str | None, title: str | None) -> None:
        job.state = "running"
        try:
            def progress(stage: str, frac: float) -> None:
                job.stage, job.progress = stage, round(frac, 3)

            meeting, samples = process_file(
                path, date, transcriber=get_transcriber(), extractor=get_extractor(extractor), diarise=diarise,
                models_dir=MODELS_DIR, title=title, progress=progress,
            )  # fmt: skip
            job.meeting = meeting
            job.peaks = audio_mod.peaks(samples)
            job.state, job.stage, job.progress = "done", "done", 1.0
        except Exception as exc:  # surfaced to the client as a readable message
            job.state, job.error = "error", str(exc)[:300]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # uploaded audio is never kept

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        from . import diarize

        return Health(
            status="ok",
            version=__version__,
            whisper_model=os.environ.get("VOICESCRIBE_WHISPER", "base.en"),
            extractor=os.environ.get("VOICESCRIBE_EXTRACTOR", "rules"),
            diarization_available=diarize.available(MODELS_DIR),
            hosted_keys={"openai": bool(os.environ.get("OPENAI_API_KEY")), "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY"))},
            max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            max_minutes=audio_mod.MAX_DURATION_S // 60,
        )

    @app.post("/api/jobs", response_model=JobStatus, status_code=202)
    async def create_job(
        file: UploadFile = File(...),
        date: str | None = Form(None),
        title: str | None = Form(None),
        diarize: bool = Form(False),
        extractor: str | None = Form(None),
    ) -> JobStatus:
        try:
            meeting_date = dt.date.fromisoformat(date) if date else dt.date.today()
        except ValueError as exc:
            raise HTTPException(422, "date must be YYYY-MM-DD") from exc
        if extractor and extractor not in {"rules", "local", "openai", "anthropic"}:
            raise HTTPException(422, "unknown extractor")
        tmp = Path(tempfile.mkdtemp(prefix="voicescribe-"))
        path = tmp / "upload.bin"
        size = 0
        try:
            with path.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                    fh.write(chunk)
            if size == 0:
                raise HTTPException(422, "empty file")
        except HTTPException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        job = Job(uuid.uuid4().hex[:12])
        with lock:
            jobs[job.id] = job
            while len(jobs) > MAX_JOBS:
                jobs.popitem(last=False)
        pool.submit(run_job, job, path, tmp, meeting_date, diarize, extractor, (title or Path(file.filename or "meeting").stem)[:80])
        return job.status()

    def find(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    @app.get("/api/jobs/{job_id}", response_model=JobResult)
    def get_job(job_id: str) -> JobResult:
        job = find(job_id)
        base = job.status().model_dump()
        return JobResult(**base, meeting=job.meeting, peaks=job.peaks)

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str) -> StreamingResponse:
        job = find(job_id)

        async def stream():
            last = None
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                cur = (job.state, job.stage, job.progress)
                if cur != last:
                    yield f"data: {json.dumps(job.status().model_dump())}\n\n"
                    last = cur
                if job.state in ("done", "error"):
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/export")
    def export_job(job_id: str, format: Literal["md", "json", "srt", "vtt"] = Query("md")) -> PlainTextResponse:
        job = find(job_id)
        if job.meeting is None:
            raise HTTPException(409, "job not finished")
        content, media = export(job.meeting, format)
        return PlainTextResponse(
            content, media_type=media, headers={"Content-Disposition": f'attachment; filename="meeting.{format}"'}
        )

    @app.exception_handler(Exception)
    async def unhandled(_request, exc: Exception) -> JSONResponse:  # never leak a traceback to the client
        return JSONResponse({"detail": "internal error"}, status_code=500)

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
