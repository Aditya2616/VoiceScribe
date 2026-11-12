# VoiceScribe: research notes

Everything marked "measured" was run on this machine (Windows 11, 12 logical cores, CPU only, shared with eight other
builds, so timings are noisy and pessimistic).

## 1. Speech recognition with word timestamps

| Option | Licence | Word timestamps | Fit here |
|---|---|---|---|
| **faster-whisper** 1.2.1 on CTranslate2 4.9 | MIT | `transcribe(..., word_timestamps=True)`, cross-attention DTW | Chosen. int8 on CPU, wheels for Py 3.13, no torch needed |
| openai-whisper | MIT | yes (same DTW) | Needs torch + an ffmpeg binary, several times slower on CPU per faster-whisper's own table |
| transformers ASR pipeline | Apache-2.0 | `return_timestamps="word"` | Slower and heavier; fallback only |
| distil-whisper (distil-large-v3) | MIT | via faster-whisper | English only, about 1.5 GB: too big for the CPU and 3 GB budget |
| WhisperX | BSD-2 | wav2vec2 forced alignment | Its diarisation uses gated pyannote checkpoints (token needed): rejected |
| whisper.cpp | MIT | yes | Needs a compiled binary, no clean Python path on Windows: rejected |

**Decoding without ffmpeg.** faster-whisper decodes through PyAV (`av`), whose wheel bundles FFmpeg libraries, so
nothing is needed on PATH. Measured gotcha: pip resolved `av` 19.0.2, which breaks faster-whisper 1.2.1
(`TypeError: open() got an unexpected keyword argument 'metadata_errors'`). Pinning `av>=15,<17` (16.1.0 installed)
fixes it; the pin goes in `pyproject.toml`. Browser recordings (WebM/Opus) also decode through PyAV; the front end
will additionally offer 16 kHz WAV encoding as a safe path.

**Measured** on an 11 s clip (JFK "ask not", public domain, from the openai/whisper tests), int8, 6 threads, word
timestamps on, CPU shared with other jobs:

| Model | Disk | Load | Decode (11 s audio) | Notes |
|---|---|---|---|---|
| tiny.en | 75 MB | 1.6 s | 0.7 s | punctuation slip; word times drift ("ask" spans 2.26 to 3.88) |
| base.en | 145 MB | 36 s cold (download) | 6.4 s | good |
| small.en | 485 MB | 34 s cold (download) | 14.4 s | best, splits the two sentences correctly |

Decode times are inflated by contention (tiny vs base is not really 9x); the ordering is the useful part. Plan:
`base.en` default, `small.en` for the evaluation numbers, `tiny.en` for fast tests. Word timestamps come from attention
alignment and can stretch over pauses, so I will clamp and smooth them (a word never ends after the next one starts)
and check timing sanity on the synthetic meeting where onsets are known.

## 2. Segmentation and speakers

- Sentences: rule-based over Whisper's punctuation on the word stream, with a pause fallback (gap above 0.8 s).
  Topics: TextTiling-style lexical cohesion over sentence windows (numpy only).
- Diarisation without a token: **sherpa-onnx** (Apache-2.0, wheel 2.0.1 installed fine) with
  `sherpa-onnx-pyannote-segmentation-3-0` (7 MB, MIT, public GitHub release asset, HEAD 200) plus an embedding model:
  `wespeaker_en_voxceleb_resnet34.onnx` (26 MB, toolkit Apache-2.0, VoxCeleb-trained weights CC-BY-4.0, credit needed) or
  NeMo TitaNet-small (40 MB, CC-BY-4.0). All are plain GitHub release URLs, no HF token. The upstream
  pyannote/speaker-diarization-3.1 pipeline is gated; Rev reverb-diarization is non-commercial. Neither is used.
- Diarisation is optional. Not yet measured: its accuracy. It will be scored on the synthetic meeting, where speaker
  turns are known exactly.

## 3. Summarisation and extraction

**Schema** (Pydantic v2): `Meeting{date, title, summary, decisions[], action_items[], speakers[]}`;
`ActionItem{task, owner|null, deadline_text|null, deadline_date|null, source{start, end, sentence_ids, quote}, confidence}`.
Validation rejects items whose span does not exist or whose quote is not in the transcript. Relative dates ("next
Friday", "by the 20th", "end of the month", "tomorrow", "in two weeks") are resolved against the meeting date by a
small rule-based resolver (python-dateutil, installed). `dateparser` 1.5.0 installs but is heavy and fuzzy, so it is
only a candidate fallback.

**Extractor ladder**, one interface, chosen by env var:
1. `rules`: deterministic baseline and fallback. Cue patterns for commitments ("I'll", "X will", "can you", "please",
   "needs to", "action item", "by <date>") and decisions ("we decided", "agreed", "let's go with", "approved");
   owner from addressee or speaker; deadline from the resolver.
2. `local`: a small instruct model through transformers. **Measured** on a 4-sentence test (greedy, bf16, CPU, contended):
   - Qwen2.5-0.5B-Instruct (Apache-2.0, about 1 GB): 14 s for 65 tokens. Valid JSON but wrong content: listed a
     decision as Tom's task, mangled "send the revised budget to finance", invented an owner.
   - Qwen2.5-1.5B-Instruct (Apache-2.0, about 3 GB): 48 s for 92 tokens (about 2 tok/s; fp32 may be faster than bf16
     here, to test). Tasks and owners correct; it still called a decision an action item with owner "team".
   Conclusion: at this size the model is a **verifier and normaliser over rule-based candidates** (classify each
   candidate sentence, rewrite the task phrase, choose the owner from a closed list of known speakers and names), not a
   free generator over the transcript. Short prompts and short outputs are also much faster, and the closed vocabulary
   prevents hallucinated owners. JSON reliability: tiny output format, forced JSON prefix, lenient parse, schema
   validation, fallback to the rules answer. Constrained-decoding libraries (outlines, lm-format-enforcer, llama.cpp
   grammars) were considered; with transformers 5.x on Python 3.13 the dependency risk outweighs the benefit.
3. `openai` and `anthropic`: adapters over `httpx` (no SDK, avoids version drift), JSON-schema structured output and
   tool use respectively; tested with `httpx.MockTransport`. Not run for real (no keys).

**Summary.** The offline default is extractive (sentence embeddings from the cached all-MiniLM-L6-v2, MIT, ranked
centrality plus position, re-ordered chronologically). The LLM adapters give abstractive summaries. The README will
say the offline summary is extractive.

## 4. Data and licences

| Data | Licence | Use |
|---|---|---|
| LibriSpeech test-clean (OpenSLR 12) | CC-BY-4.0 | WER set: about 12 utterances, 3+ speakers, a few MB of flac committed with credit; reference is the official transcript. Fetched from openslr.org (346 MB archive, transient), only chosen files kept |
| AMI Meeting Corpus | CC-BY-4.0 for signals and transcripts (AMI download page) | Real meeting audio but large, far-field and has no action-item labels. Not bundled; optional documented script at most |
| ICSI Meeting Corpus | believed CC-BY-4.0, not verified | not used |
| MeetingBank | HF mirrors tag CC-BY-NC-SA-4.0, no official statement found | not used (incompatible with an MIT repo) |
| Kokoro-82M via kokoro-onnx 0.6.1 (int8 model 92 MB, voices 28 MB) | Apache-2.0 | Synthesises the multi-speaker demo meeting. Measured: works, about 1.15x slower than real time, 54 voices. Phonemisation calls espeak-ng (GPL) as a separate tool; the generated audio is ours |
| Hand-written meeting transcripts with labels | mine, MIT | Precision/recall set: about 8 meetings of 12-25 sentences, labelled for action items (task, owner, deadline) and decisions |
| JFK clip | US government work, public domain | smoke test only |

**The demo meeting is synthetic.** No permissively licensed, clean-audio meeting recording of a suitable length is
practical to bundle, so I write a script and voice it with Kokoro. It is labelled synthetic in the UI and README. It
has exact ground truth for words, speakers and action items. WER on real human speech comes from the LibriSpeech
subset, which is read speech, not conversational meeting speech; the README must say that meeting-style WER is not
measured.

## 5. Related open source
faster-whisper, WhisperX, whisper.cpp, stable-ts, whisper-timestamped; pyannote.audio and sherpa-onnx for diarisation;
the MeetingBank paper and AMI-based summarisation work; Outlines and llama.cpp grammars for constrained JSON.

## 6. Sources
- https://github.com/SYSTRAN/faster-whisper and https://pypi.org/project/faster-whisper
- https://github.com/k2-fsa/sherpa-onnx and https://huggingface.co/OpenASR/pyannote-segmentation-3.0 (MIT mirror)
- https://groups.inf.ed.ac.uk/ami/download (CC-BY-4.0)
- https://arxiv.org/abs/2305.17529 (MeetingBank) and its HF dataset cards
- https://github.com/thewh1teagle/kokoro-onnx; Qwen2.5 model cards (Apache-2.0)
- https://www.openslr.org/12 (LibriSpeech, CC-BY-4.0)
