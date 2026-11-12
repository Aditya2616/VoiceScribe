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
EMBEDDING_DIR = f"{BASE}/speaker-recongition-models"
SEGMENTATION_EMBEDDING = "wespeaker_en_voxceleb_resnet34"  # used by the segmentation pipeline baseline
DEFAULT_EMBEDDING = "nemo_en_titanet_small"  # used by sentence-level diarisation (chosen on the dev meetings)
SENTENCE_DEFAULTS = {"embedding": DEFAULT_EMBEDDING, "threshold": 0.35, "min_seconds": 1.0}  # see results/diarize_chosen.json
EMBEDDINGS = [  # speaker embedding models published as sherpa-onnx release assets
    "wespeaker_en_voxceleb_resnet34", "wespeaker_en_voxceleb_CAM++", "wespeaker_en_voxceleb_CAM++_LM", "nemo_en_titanet_small",
]  # fmt: skip


def model_paths(models_dir: Path, int8: bool = False, embedding: str = DEFAULT_EMBEDDING) -> tuple[Path, Path]:
    folder = models_dir / "sherpa-onnx-pyannote-segmentation-3-0"
    seg = folder / ("model.int8.onnx" if int8 else "model.onnx")
    if not seg.exists():
        seg = folder / ("model.onnx" if int8 else "model.int8.onnx")
    return seg, models_dir / f"{embedding}.onnx"


def available(models_dir: Path, embedding: str = DEFAULT_EMBEDDING) -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return model_paths(models_dir, embedding=embedding)[1].exists()


def download(models_dir: Path, embedding: str = DEFAULT_EMBEDDING, with_segmentation: bool = False) -> None:
    """Fetch the speaker embedding model (about 40 MB). The segmentation model is only for the baseline pipeline."""
    models_dir.mkdir(parents=True, exist_ok=True)
    seg, emb = model_paths(models_dir, embedding=embedding)
    if with_segmentation and not seg.exists():
        archive = models_dir / "seg.tar.bz2"
        urllib.request.urlretrieve(SEGMENTATION_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(models_dir, filter="data")
        archive.unlink()
    if not emb.exists():
        urllib.request.urlretrieve(f"{EMBEDDING_DIR}/{embedding}.onnx", emb)


def diarize(
    samples: np.ndarray, models_dir: Path, num_speakers: int = -1, threshold: float = 0.5,
    min_on: float = 0.3, min_off: float = 0.4, int8: bool = False, threads: int | None = None,
    embedding: str = SEGMENTATION_EMBEDDING,
):  # fmt: skip
    """Return [(start, end, "Speaker 1"), ...] sorted by start. Speakers are numbered by first appearance."""
    import sherpa_onnx

    seg, emb = model_paths(models_dir, int8, embedding)
    threads = threads or max(1, default_threads() - 1)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=min_on,
        min_duration_off=min_off,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    result = sd.process(samples.astype(np.float32)).sort_by_start_time()
    order: dict[int, str] = {}
    turns = []
    for r in result:
        label = order.setdefault(r.speaker, f"Speaker {len(order) + 1}")
        turns.append((float(r.start), float(r.end), label))
    return turns


# ---- sentence-level diarisation -------------------------------------------------------------------
# The segmentation pipeline above works on arbitrary slices of audio. Meetings are made of short sentences, each said
# by one person, so a second method embeds each recognised sentence as a whole and clusters the embeddings. It is
# also quicker and lets the number of speakers (for example the length of a participants list) steer the clustering.


def embed_spans(samples: np.ndarray, spans: list[tuple[float, float]], models_dir: Path,
                embedding: str = DEFAULT_EMBEDDING, threads: int | None = None) -> np.ndarray:  # fmt: skip
    """One speaker embedding per (start, end) span, with the sherpa-onnx extractor."""
    import sherpa_onnx

    from .audio import SAMPLE_RATE

    _, emb = model_paths(models_dir, embedding=embedding)
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads or max(1, default_threads() - 1))
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    out = []
    for a, b in spans:
        chunk = samples[int(a * SAMPLE_RATE) : max(int(b * SAMPLE_RATE), int(a * SAMPLE_RATE) + 1600)].astype(np.float32)
        st = ex.create_stream()
        st.accept_waveform(sample_rate=SAMPLE_RATE, waveform=chunk)
        st.input_finished()
        out.append(np.asarray(ex.compute(st), dtype=np.float64))
    return np.vstack(out) if out else np.zeros((0, 1))


def cluster_spans(emb: np.ndarray, durations: list[float], k: int | None = None, threshold: float = 0.5,
                  min_seconds: float = 1.0) -> list[int]:  # fmt: skip
    """Average-linkage clustering (cosine distance) of span embeddings. Spans shorter than `min_seconds` are too
    short to trust: they do not shape the clusters and join the nearest cluster afterwards. With `k` the clustering
    stops at k clusters, otherwise at cosine distance `threshold`."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(emb)
    if n == 0:
        return []
    long = [i for i in range(n) if durations[i] >= min_seconds] or list(range(n))
    if len(long) == 1:
        return [0] * n
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    tree = linkage(norm[long], method="average", metric="cosine")
    lab = fcluster(tree, t=k, criterion="maxclust") if k else fcluster(tree, t=threshold, criterion="distance")
    labels = [-1] * n
    for idx, c in zip(long, lab, strict=True):
        labels[idx] = int(c)
    centroids = {c: norm[[i for i in long if labels[i] == c]].mean(axis=0) for c in set(lab)}
    for i in range(n):
        if labels[i] < 0:
            labels[i] = max(centroids, key=lambda c: float(norm[i] @ centroids[c]))
    return labels


def diarize_sentences(samples: np.ndarray, spans: list[tuple[float, float]], models_dir: Path, num_speakers: int | None = None,
                      threshold: float = 0.5, embedding: str = DEFAULT_EMBEDDING, threads: int | None = None,
                      min_seconds: float = 1.0):  # fmt: skip
    """Turns [(start, end, "Speaker N")] with one turn per sentence span, speakers numbered by first appearance."""
    emb = embed_spans(samples, spans, models_dir, embedding, threads)
    labels = cluster_spans(emb, [b - a for a, b in spans], num_speakers, threshold, min_seconds)
    order: dict[int, str] = {}
    return [(a, b, order.setdefault(lab, f"Speaker {len(order) + 1}")) for (a, b), lab in zip(spans, labels, strict=True)]
