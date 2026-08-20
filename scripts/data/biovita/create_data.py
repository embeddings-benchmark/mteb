from __future__ import annotations

from collections import defaultdict

import json
import logging
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

SOURCE_PATH = "risashinoda/BioVITA"
SOURCE_REVISION = "6d8ad01ac05228738c2f9b88e3670465017f2ea3"
GROUP = "unseen"
SUBSET_LEVELS = {
    "unseen_species": "species",
    "unseen_genus": "genus",
}

FALLBACK_SAMPLE_RATE = 48000
FALLBACK_SECONDS = 10.0

logger = logging.getLogger(__name__)


@cache
def download_csv(relative_path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            SOURCE_PATH,
            f"{GROUP}/{relative_path}",
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
    )


@cache
def download_media(modality: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=SOURCE_PATH,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=[f"{GROUP}/{modality}/**"],
        )
    )


def _resolve_audio_paths(
    file_paths: list[str], species: list[str], root: Path
) -> list[Path]:
    """Map `audio_index.csv` rows onto the audio files published in the repo.

    `audio_index.csv` does not name the published files consistently: only 736
    of its 1024 `file_path` values exist verbatim. 258 name an Animal Sound
    Archive clip as `asa_<recording_id>.mp3` while the repo stores it under a
    descriptive archive filename, and 30 name a xeno-canto clip by its original
    upload filename (e.g. `White-crowned_Koel_WProv_IW5.mp3`) while the repo
    stores it as `xc_<id>.mp3`. The published names follow `download_audio.py`,
    which writes `{source}_{recording_id}`; the index kept the pre-download
    names.

    The join is recovered from row order instead of from filenames:
    `audio_index.csv` and `metadata.csv` are two views of one table, so within a
    species the k-th index row is the k-th metadata row, and `metadata.file_name`
    gives the published file. This is not a similarity heuristic and not an
    arbitrary ordering -- it is checked below against every row whose file is
    independently identifiable (the 736 verbatim names and the 258 embedded
    recording ids, 994 rows in total), and `_verify_audio_mapping` raises if any
    of them disagrees. Row order is meaningful here: the index is *not* in
    metadata order globally (only 767/1024 rows line up), and only 48 of 325
    species happen to be in filename-sorted order, so this correspondence
    carries real information rather than restating a sort.
    """
    import pandas as pd

    metadata = pd.read_csv(download_csv("metadata.csv"))
    files_by_species: dict[str, list[Path]] = defaultdict(list)
    for name, taxon in zip(metadata["file_name"], metadata["scientific_name"]):
        files_by_species[str(taxon)].append(root / GROUP / str(name))

    taken: dict[str, int] = defaultdict(int)
    resolved: list[Path] = []
    for taxon in species:
        pool = files_by_species.get(taxon, [])
        position = taken[taxon]
        taken[taxon] += 1
        if position >= len(pool):
            raise ValueError(
                f"BioVITA audio index lists more {taxon!r} rows than metadata.csv publishes"
            )
        resolved.append(pool[position])

    _verify_audio_mapping(file_paths, resolved, root)
    return resolved


def _verify_audio_mapping(
    file_paths: list[str], resolved: list[Path], root: Path
) -> None:
    """Check the row-order mapping against every independently identifiable row."""
    checked = 0
    for relative, path in zip(file_paths, resolved):
        if (root / GROUP / relative).exists():
            if path != root / GROUP / relative:
                raise ValueError(
                    f"BioVITA audio mapping disagrees for {relative!r}: got {path.name}"
                )
            checked += 1
            continue
        named = re.match(r"^(?:xc|asa|inat)_(\d+)$", Path(relative).stem, re.IGNORECASE)
        if named:
            if named.group(1) not in re.findall(r"\d+", path.name):
                raise ValueError(
                    f"BioVITA audio mapping disagrees for {relative!r}: "
                    f"recording id {named.group(1)} absent from {path.name}"
                )
            checked += 1
    if len(set(resolved)) != len(resolved):
        raise ValueError("BioVITA audio mapping is not one-to-one")
    if not checked:
        # Guards against a future revision in which no row is independently
        # identifiable, which would let the row-order join pass vacuously.
        raise ValueError(
            "BioVITA audio mapping could not be checked against any row: no index "
            "filename matches a published file or carries a recording id"
        )
    logger.debug(
        "BioVITA: audio mapping verified on %d/%d independently identifiable rows",
        checked,
        len(file_paths),
    )


def _decodes(path: Path) -> bool:
    try:
        from torchcodec.decoders import AudioDecoder
    except ImportError:  # no ffmpeg-backed decoder installed; nothing to check
        return True
    try:
        AudioDecoder(str(path)).get_all_samples()
    except Exception:
        return False
    return True


def _wav_bytes(samples: Any, sampling_rate: int) -> bytes:
    """In-memory 16-bit PCM wav for an already decoded waveform."""
    import io
    import wave

    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767).astype(
        "<i2"
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


@cache
def audio_fallbacks(paths: tuple[Path, ...]) -> tuple[dict[str, Any], ...] | None:
    """Reproduce the official decode fallback for clips `datasets` cannot read.

    The official `load_audio_onset_10s` decodes with `torchaudio`, falls back to
    `librosa` on failure, and finally returns a zero waveform. MTEB decodes
    through `datasets.Audio` (torchcodec/ffmpeg) alone, which rejects ten of the
    1024 published clips: nine xeno-canto mp3s carry trailing garbage after the
    audio stream, and `Porzana_pusilla_..._V0266_19_short.mp3` is an HTML error
    page saved under an `.mp3` name. For exactly those clips the same
    librosa-then-silence fallback is applied here and handed to `datasets.Audio`
    as in-memory bytes, so the model sees what the reference implementation sees.
    The published files are read-only and are never modified or re-encoded on
    disk. Returns None when every clip decodes normally.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=16) as pool:
        decodes = list(pool.map(_decodes, paths))
    if all(decodes):
        return None

    payloads: list[dict[str, Any]] = []
    for path, ok in zip(paths, decodes):
        if ok:
            payloads.append({"bytes": None, "path": str(path)})
            continue
        try:
            import librosa

            samples, sampling_rate = librosa.load(str(path), sr=None, mono=True)
        except Exception:
            import numpy as np

            samples = np.zeros(
                int(FALLBACK_SAMPLE_RATE * FALLBACK_SECONDS), dtype="float32"
            )
            sampling_rate = FALLBACK_SAMPLE_RATE
            logger.warning(
                "BioVITA: %s cannot be decoded at all and is scored as silence, "
                "matching the official evaluation script's zero-waveform fallback",
                path.name,
            )
        payloads.append(
            {"bytes": _wav_bytes(samples, int(sampling_rate)), "path": str(path)}
        )
def load_index(modality: str, level: str) -> tuple[Any, list[str], list[str]]:
    """One modality index as a HF dataset of `id` plus the payload column.

    Cached across tasks and subsets: the same audio and image indices back all
    six retrieval directions, and rebuilding them per split is pure overhead.
    """
    import pandas as pd
    from datasets import Audio, Dataset, Image

    index = pd.read_csv(download_csv(f"benchmark/{modality}_index.csv"))
    ids = [str(i) for i in index["id"]]
    taxa = [str(t) for t in index[level]]

    if modality == "text":
        # The official feature extractor embeds the bare taxon name of the
        # evaluated level, so genus-level text repeats across the species of a
        # genus. Those duplicates stay separate documents because they are
        # separate ids in the official candidate groups.
        return Dataset.from_dict({"id": ids, "text": taxa}), ids, taxa

    root = download_media(modality)
    if modality == "image":
        paths = [str(root / GROUP / path) for path in index["file_path"]]
        column, feature = "image", Image()
    else:
        resolved = _resolve_audio_paths(
            [str(path) for path in index["file_path"]],
            [str(name) for name in index["species"]],
            root,
        )
        fallbacks = audio_fallbacks(tuple(resolved))
        paths = (
            [str(path) for path in resolved] if fallbacks is None else list(fallbacks)
        )
        column, feature = "audio", Audio()

    data = Dataset.from_dict({"id": ids, column: paths}).cast_column(column, feature)
    return data, ids, taxa


def build_split(
    query_modality: str,
    document_modality: str,
    csv_name: str,
    level: str,
    *,
    metadata_only: bool = False,
):
    import pandas as pd
    from datasets import Dataset

    if metadata_only:
        corpus_raw = pd.read_csv(
            download_csv(f"benchmark/{document_modality}_index.csv")
        )
        query_raw = pd.read_csv(
            download_csv(f"benchmark/{query_modality}_index.csv")
        )

        corpus_ids = [str(i) for i in corpus_raw["id"]]
        corpus_taxa = [str(t) for t in corpus_raw[level]]
        query_index_ids = [str(i) for i in query_raw["id"]]

        corpus = Dataset.from_dict(
            {
                "id": corpus_ids,
                "taxon": corpus_taxa,
            }
        )
        query_index = Dataset.from_dict(
            {
                "id": query_index_ids,
                "text": [str(t) for t in query_raw[level]],
            }
        )
    else:
        corpus, corpus_ids, corpus_taxa = load_index(document_modality, level)
        query_index, query_index_ids, _ = load_index(query_modality, level)
        corpus = corpus.add_column("taxon", corpus_taxa)

    tasks = pd.read_csv(download_csv(f"benchmark/{level}/{csv_name}"))
    query_ids = [str(query_id) for query_id in tasks["query_id"]]

    row_of_payload = {
        payload: row for row, payload in enumerate(query_index_ids)
    }

    queries = (
        query_index.select(
            [row_of_payload[str(payload)] for payload in tasks["query_payload_id"]]
        )
        .remove_columns("id")
        .add_column("id", query_ids)
    )

    candidate_taxa = []
    correct_taxa = []
    qrels = []
    top_ranked = []

    for query_id, correct, taxa_json, ids_json in zip(
        query_ids,
        tasks["correct_taxon"],
        tasks["candidates_taxa"],
        tasks["candidates_target_ids"],
    ):
        taxa = [str(taxon) for taxon in json.loads(taxa_json)]
        groups = [
            [str(int(i)) for i in group]
            for group in json.loads(ids_json)
        ]

        correct = str(correct)
        candidate_taxa.append(taxa)
        correct_taxa.append(correct)

        for doc_id in groups[taxa.index(correct)]:
            qrels.append(
                {
                    "query-id": query_id,
                    "corpus-id": doc_id,
                    "score": 1,
                }
            )

        top_ranked.append(
            {
                "query-id": query_id,
                "corpus-ids": [
                    doc_id for group in groups for doc_id in group
                ],
            }
        )

    queries = queries.add_column("correct_taxon", correct_taxa)
    queries = queries.add_column("candidate_taxa", candidate_taxa)

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": Dataset.from_list(qrels),
        "top_ranked": Dataset.from_list(top_ranked),
    }


TASKS = {
    "BioVITAA2TRetrieval": {
        "query_modality": "audio",
        "document_modality": "text",
        "csv_name": "test_audio_to_text.csv",
    },
    "BioVITAT2ARetrieval": {
        "query_modality": "text",
        "document_modality": "audio",
        "csv_name": "test_text_to_audio.csv",
    },
    "BioVITAA2IRetrieval": {
        "query_modality": "audio",
        "document_modality": "image",
        "csv_name": "test_audio_to_image.csv",
    },
    "BioVITAI2ARetrieval": {
        "query_modality": "image",
        "document_modality": "audio",
        "csv_name": "test_image_to_audio.csv",
    },
    "BioVITAI2TRetrieval": {
        "query_modality": "image",
        "document_modality": "text",
        "csv_name": "test_image_to_text.csv",
    },
    "BioVITAT2IRetrieval": {
        "query_modality": "text",
        "document_modality": "image",
        "csv_name": "test_text_to_image.csv",
    },
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=sorted(TASKS),
        help="BioVITA MTEB task to construct.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Build metadata without downloading image/audio media.",
    )
    parser.add_argument(
        "--repo-id",
        help="Hugging Face dataset repo ID.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the constructed fixed-format dataset to the Hub.",
    )
    args = parser.parse_args()

    if args.push and args.metadata_only:
        raise SystemExit("--metadata-only cannot be used with --push")
    if args.push and not args.repo_id:
        raise SystemExit("--repo-id is required with --push")

    task = TASKS[args.task]

    for subset, level in SUBSET_LEVELS.items():
        data = build_split(
            query_modality=task["query_modality"],
            document_modality=task["document_modality"],
            csv_name=task["csv_name"],
            level=level,
            metadata_only=args.metadata_only,
        )

        print(f"\n== {args.task} / {subset} ==")
        for name, dataset in data.items():
            print(name, len(dataset), dataset.features)

        if args.push:
            from datasets import DatasetDict
            from huggingface_hub import create_repo

            token = os.environ.get("HF_TOKEN")

            create_repo(
                args.repo_id,
                repo_type="dataset",
                token=token,
                exist_ok=True,
            )

            for name, dataset in data.items():
                config_name = f"{subset}-{name}"
                DatasetDict({"test": dataset}).push_to_hub(
                    args.repo_id,
                    config_name,
                    token=token,
                )
                print(f"Pushed {config_name}")

    if args.push:
        print(f"Pushed {args.repo_id}. Pin the commit SHA in TaskMetadata.")


if __name__ == "__main__":
    main()
