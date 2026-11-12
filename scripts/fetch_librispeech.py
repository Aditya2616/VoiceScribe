"""Select a small, fixed LibriSpeech test-clean subset (CC-BY-4.0) for the WER check.

Streams the OpenSLR archive and stops as soon as it has N utterances from each of the first K speakers
in archive order, so only the beginning of the 346 MB file is read. Writes flac + refs.tsv to
data/eval/wer/. Licence: CC-BY-4.0, Panayotov et al., "LibriSpeech: an ASR corpus based on public domain
audio books", ICASSP 2015.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "wer"
SPEAKERS, PER_SPEAKER = 4, 3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    refs: dict[str, str] = {}
    flacs: dict[str, bytes] = {}
    speakers: list[str] = []
    chosen: dict[str, list[str]] = defaultdict(list)
    with urllib.request.urlopen(URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            parts = member.name.split("/")
            if len(parts) < 4:
                continue
            speaker = parts[-3]
            data = tar.extractfile(member).read()
            if name.endswith(".trans.txt"):
                for line in data.decode().splitlines():
                    uid, _, text = line.partition(" ")
                    refs[uid] = text
            elif name.endswith(".flac"):
                uid = name[:-5]
                if speaker not in speakers:
                    if len(speakers) >= SPEAKERS:
                        continue
                    speakers.append(speaker)
                if len(chosen[speaker]) < PER_SPEAKER and 4_000 < len(data) < 400_000:
                    chosen[speaker].append(uid)
                    flacs[uid] = data
            done = len(speakers) >= SPEAKERS and all(len(chosen[s]) >= PER_SPEAKER for s in speakers)
            if done and all(u in refs for s in speakers for u in chosen[s]):
                break
    rows = []
    for s in speakers:
        for uid in chosen[s]:
            (OUT / f"{uid}.flac").write_bytes(flacs[uid])
            rows.append(f"{uid}\t{refs[uid]}")
    (OUT / "refs.tsv").write_text("\n".join(rows) + "\n", encoding="utf8")
    total = sum(len(v) for v in flacs.values())
    print(f"wrote {len(rows)} utterances, {total / 1e6:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
