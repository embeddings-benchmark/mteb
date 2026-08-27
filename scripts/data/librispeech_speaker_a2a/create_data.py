"""Build the LibriSpeechSpeakerA2ARetrieval audio-to-audio retrieval task
from LibriSpeech test-clean.

Source: openslr/librispeech_asr (config "clean", split "test"), the
standard LibriSpeech test-clean set (2,620 utterances from 40 speakers,
CC-BY-4.0). Unlike FSD50KA2ARetrieval (repurposed from a classification
task), this is a genuinely native audio-to-audio retrieval formulation:
given a speech clip (query), retrieve other clips from the same speaker
(a real task family -- speaker re-identification/verification -- distinct
from LibriTTS's existing a2t/t2a transcription-retrieval tasks in mteb,
which test text-audio matching, not speaker identity).

Every one of LibriSpeech test-clean's 40 speakers has at least 32 clips,
well above the QUERIES_PER_SPEAKER + CORPUS_PER_SPEAKER threshold, so all
40 are used (unlike FSD50K's 20-of-many-classes selection, there's no need
to drop speakers here). Per speaker, QUERIES_PER_SPEAKER clips become
queries and a disjoint CORPUS_PER_SPEAKER clips become corpus documents.
Relevance (qrels) is same-speaker membership: every query is relevant to
every corpus document from the same speaker.

    200 queries + 400 corpus docs (40 speakers x 5 queries x 10 corpus)
    2000 qrels rows (40 speakers x 5 queries x 10 corpus docs each)

Usage:
    python scripts/data/librispeech_speaker_a2a/create_data.py \\
        --push-to-hub yaswanth169/LibriSpeech-Speaker-A2ARetrieval
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from datasets import Audio, Dataset, DatasetDict, load_dataset

SOURCE_DATASET = "openslr/librispeech_asr"
SOURCE_CONFIG = "clean"
SOURCE_SPLIT = "test"
QUERIES_PER_SPEAKER = 5
CORPUS_PER_SPEAKER = 10
SEED = 42


def build_splits(seed: int = SEED) -> tuple[Dataset, Dataset, list[dict]]:
    source = load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split=SOURCE_SPLIT)

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for idx, speaker_id in enumerate(source["speaker_id"]):
        by_speaker[speaker_id].append(idx)

    min_per_speaker = QUERIES_PER_SPEAKER + CORPUS_PER_SPEAKER
    speakers = sorted(
        spk for spk, idxs in by_speaker.items() if len(idxs) >= min_per_speaker
    )
    assert len(speakers) == len(by_speaker), (
        f"only {len(speakers)} of {len(by_speaker)} speakers have "
        f">= {min_per_speaker} clips"
    )

    rng = random.Random(seed)
    query_rows: list[dict] = []
    corpus_rows: list[dict] = []
    qrels: list[dict] = []
    for speaker in speakers:
        idxs = list(by_speaker[speaker])
        rng.shuffle(idxs)
        query_idxs = idxs[:QUERIES_PER_SPEAKER]
        corpus_idxs = idxs[
            QUERIES_PER_SPEAKER : QUERIES_PER_SPEAKER + CORPUS_PER_SPEAKER
        ]

        query_ids = [f"q-{speaker}-{i}" for i in range(len(query_idxs))]
        corpus_ids = [f"c-{speaker}-{i}" for i in range(len(corpus_idxs))]

        for qid, idx in zip(query_ids, query_idxs):
            query_rows.append({"id": qid, "audio": source[idx]["audio"]})
        for cid, idx in zip(corpus_ids, corpus_idxs):
            corpus_rows.append({"id": cid, "audio": source[idx]["audio"]})
        for qid in query_ids:
            for cid in corpus_ids:
                qrels.append({"query-id": qid, "corpus-id": cid, "score": 1})

    # Dataset.from_list doesn't infer the Audio() feature type from raw
    # decoded audio objects -- cast explicitly or the pushed dataset's
    # audio column comes back as a plain {bytes, path} struct with no
    # sampling_rate, which mteb's audio collator requires.
    queries = Dataset.from_list(query_rows).cast_column("audio", Audio())
    corpus = Dataset.from_list(corpus_rows).cast_column("audio", Audio())
    return queries, corpus, qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", default=None, help="HF repo id to push to")
    args = parser.parse_args()

    queries, corpus, qrels = build_splits()
    qrels_ds = Dataset.from_list(qrels)
    num_speakers = len(queries) // QUERIES_PER_SPEAKER

    print(f"queries: {len(queries)}, corpus: {len(corpus)}, qrels: {len(qrels_ds)}")
    assert len(queries) == num_speakers * QUERIES_PER_SPEAKER
    assert len(corpus) == num_speakers * CORPUS_PER_SPEAKER
    assert len(qrels_ds) == num_speakers * QUERIES_PER_SPEAKER * CORPUS_PER_SPEAKER
    assert set(queries["id"]).isdisjoint(set(corpus["id"]))

    if args.push_to_hub:
        DatasetDict({"test": queries}).push_to_hub(args.push_to_hub, "queries")
        DatasetDict({"test": corpus}).push_to_hub(args.push_to_hub, "corpus")
        DatasetDict({"test": qrels_ds}).push_to_hub(args.push_to_hub, "qrels")


if __name__ == "__main__":
    main()
