"""Rebuild the Clotho v2 retrieval data with the official captions.

The earlier data split captions on periods, so a caption without a final period
was glued to the next one. This keeps its audio and clip ids and takes the five
captions of each clip from the official captions file:

    python scripts/data/clotho_v2/create_data.py
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request

import datasets

CAPTIONS_URL = (
    "https://zenodo.org/records/4783391/files/clotho_captions_validation.csv?download=1"
)
A2T = ("lxercode/clotho_a2t_v2", "fb4fbd5c1e612be4894a478bd45500eb0bcd689f")
T2A = ("lxercode/clotho_t2a_v2", "13cf5105f96e1844ee3b0b84cfba9103a7ebe6a2")


def words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]", " ", text.lower()).split())


def load(repo: tuple[str, str], config: str) -> datasets.Dataset:
    return datasets.load_dataset(repo[0], config, revision=repo[1], split="test")


def main() -> None:
    with urllib.request.urlopen(CAPTIONS_URL) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode())))
    # glued captions keep their words, so a clip's joined captions find its official row
    official = {}
    for row in rows:
        captions = [row[f"caption_{k}"].strip() for k in range(1, 6)]
        official[words(" ".join(captions))] = captions
    assert len(official) == len(rows)

    old_corpus, audio = load(A2T, "corpus"), load(A2T, "queries")
    old_captions = {}
    for doc_id, text in zip(old_corpus["id"], old_corpus["text"]):
        clip, k = doc_id.removeprefix("d-").rsplit("-", 1)
        old_captions.setdefault(clip, {})[int(k)] = text
    clips = audio["id"]
    captions = {}
    for clip in clips:
        old = old_captions[clip]
        captions[clip] = official[words(" ".join(old[k] for k in sorted(old)))]

    clip_of = [clip for clip in clips for _ in range(5)]
    caption_ids = [f"{clip}-{k}" for clip in clips for k in range(5)]
    doc_ids = [f"d-{i}" for i in caption_ids]
    query_ids = [f"q-{i}" for i in caption_ids]
    texts = [text for clip in clips for text in captions[clip]]
    ones = [1] * len(texts)

    a2t = {
        "corpus": datasets.Dataset.from_dict({"id": doc_ids, "text": texts}),
        "queries": audio,
        "qrels": datasets.Dataset.from_dict(
            {
                "query-id": clip_of,
                "corpus-id": doc_ids,
                "score": ones,
            }
        ),
    }
    t2a = {
        "corpus": load(T2A, "corpus"),
        "queries": datasets.Dataset.from_dict({"id": query_ids, "text": texts}),
        "qrels": datasets.Dataset.from_dict(
            {
                "query-id": query_ids,
                "corpus-id": clip_of,
                "score": ones,
            }
        ),
    }
    for repo, configs in [("mteb/clotho_a2t_v2", a2t), ("mteb/clotho_t2a_v2", t2a)]:
        for config, ds in configs.items():
            ds.push_to_hub(repo, config_name=config, split="test")


if __name__ == "__main__":
    main()
