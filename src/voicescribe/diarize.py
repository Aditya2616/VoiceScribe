"""Optional speaker diarisation with sherpa-onnx (pyannote segmentation 3.0 ONNX + a speaker embedding model).

Both models are public GitHub release assets (no Hugging Face token). Off by default; see `fetch-models`.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .audio import default_threads

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = f"{BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMBEDDING_URL = f"{BASE}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx"


def model_paths(models_dir: Path) -> tuple[Path, Path]:
    folder = models_dir / "sherpa-onnx-pyannote-segmentation-3-0"
    seg = folder / "model.onnx"
    if not seg.exists():
        seg = folder / "model.int8.onnx"
    return seg, models_dir / "wespeaker_en_voxceleb_resnet34.onnx"


def available(models_dir: Path) -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return all(p.exists() for p in model_paths(models_dir))


def download(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    seg, emb = model_paths(models_dir)
    if not seg.exists():
        archive = models_dir / "seg.tar.bz2"
        urllib.request.urlretrieve(SEGMENTATION_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(models_dir, filter="data")
        archive.unlink()
    if not emb.exists():
        urllib.request.urlretrieve(EMBEDDING_URL, emb)


def diarize(samples: np.ndarray, models_dir: Path, num_speakers: int = -1, threshold: float = 0.5):
    """Return [(start, end, "Speaker 1"), ...] sorted by start. Speakers are numbered by first appearance."""
    import sherpa_onnx

    seg, emb = model_paths(models_dir)
    threads = max(1, default_threads() - 1)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.4,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = sd.process(samples.astype(np.float32)).sort_by_start_time()
    order: dict[int, str] = {}
    turns = []
    for r in result:
        label = order.setdefault(r.speaker, f"Speaker {len(order) + 1}")
        turns.append((float(r.start), float(r.end), label))
    return turns
