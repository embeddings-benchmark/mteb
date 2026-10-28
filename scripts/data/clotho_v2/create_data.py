"""Rebuild the Clotho v2 retrieval datasets with the official captions.

The earlier v2 data split captions on periods, so a caption without a final
period was glued to the next one (545 of 5,225). This keeps the audio and clip
ids of the earlier data and takes the five captions of each clip verbatim from
the official `clotho_captions_validation.csv` (https://zenodo.org/records/4783391):

    python scripts/data/clotho_v2/create_data.py --repo-prefix mteb/clotho
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from collections import defaultdict

import datasets

CAPTIONS_URL = (
    "https://zenodo.org/records/4783391/files/clotho_captions_validation.csv?download=1"
)
A2T = ("lxercode/clotho_a2t_v2", "fb4fbd5c1e612be4894a478bd45500eb0bcd689f")
T2A = ("lxercode/clotho_t2a_v2", "13cf5105f96e1844ee3b0b84cfba9103a7ebe6a2")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _load(repo: tuple[str, str], config: str) -> datasets.Dataset:
    return datasets.load_dataset(repo[0], config, revision=repo[1], split="test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-prefix", default="mteb/clotho")
    args = parser.parse_args()

    with urllib.request.urlopen(CAPTIONS_URL) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))
    official: dict[str, list[list[str]]] = defaultdict(list)
    for row in rows:
        captions = [row[f"caption_{k}"].strip() for k in range(1, 6)]
        official[_norm(" ".join(captions))].append(captions)

    # The glued captions still contain the same words in the same order, so the
    # joined captions of a clip identify its official row.
    a2t_corpus, a2t_queries = _load(A2T, "corpus"), _load(A2T, "queries")
    by_clip = defaultdict(list)
    for doc_id, text in zip(a2t_corpus["id"], a2t_corpus["text"]):
        clip, k = doc_id.removeprefix("d-").rsplit("-", 1)
        by_clip[clip].append((int(k), text))
    clips = list(a2t_queries["id"])
    captions = {}
    for clip in clips:
        matches = official[_norm(" ".join(t for _, t in sorted(by_clip[clip])))]
        assert len(matches) == 1, f"{clip}: {len(matches)} matching rows"
        captions[clip] = matches[0]

    docs = [(f"d-{c}-{k}", t) for c in clips for k, t in enumerate(captions[c])]
    queries = [(f"q-{c}-{k}", t) for c in clips for k, t in enumerate(captions[c])]
    t2a_corpus = _load(T2A, "corpus")
    assert sorted(t2a_corpus["id"]) == sorted(clips)

    splits = {
        "a2t": {
            "corpus": datasets.Dataset.from_dict(
                {"id": [d for d, _ in docs], "text": [t for _, t in docs]}
            ),
            "queries": a2t_queries,
            "qrels": datasets.Dataset.from_dict(
                {
                    "query-id": [
                        d.removeprefix("d-").rsplit("-", 1)[0] for d, _ in docs
                    ],
                    "corpus-id": [d for d, _ in docs],
                    "score": [1] * len(docs),
                }
            ),
        },
        "t2a": {
            "corpus": t2a_corpus,
            "queries": datasets.Dataset.from_dict(
                {"id": [q for q, _ in queries], "text": [t for _, t in queries]}
            ),
            "qrels": datasets.Dataset.from_dict(
                {
                    "query-id": [q for q, _ in queries],
                    "corpus-id": [
                        q.removeprefix("q-").rsplit("-", 1)[0] for q, _ in queries
                    ],
                    "score": [1] * len(queries),
                }
            ),
        },
    }
    for name, configs in splits.items():
        for config, ds in configs.items():
            ds.push_to_hub(
                f"{args.repo_prefix}_{name}_v2", config_name=config, split="test"
            )


if __name__ == "__main__":
    main()
