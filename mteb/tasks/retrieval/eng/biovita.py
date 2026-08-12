from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

_PATH = "risashinoda/BioVITA"
_REVISION = "6d8ad01ac05228738c2f9b88e3670465017f2ea3"
_REFERENCE = "https://arxiv.org/abs/2603.23883"

# Only the `unseen` half of the official benchmark is released. `test/benchmark`
# (the "seen" half) ships the image and text indices but neither
# `audio_index.csv` nor the `species|genus|family/test_*.csv` task files, so it
# cannot be reconstructed. `unseen/benchmark/family/*.csv` are header-only
# upstream, so only the species and genus levels are exposed here.
_GROUP = "unseen"
_SUBSET_LEVELS = {"unseen_species": "species", "unseen_genus": "genus"}

# The official `load_audio_onset_10s` returns a 10 s zero waveform at 48 kHz when
# a clip cannot be decoded at all. The same shape is used for that fallback here
# so an undecodable clip is reported identically to the reference pipeline.
_FALLBACK_SAMPLE_RATE = 48000
_FALLBACK_SECONDS = 10.0

_BIBTEX = r"""
@inproceedings{shinoda2026biovita,
  author = {Shinoda, Risa and Shiohara, Kaede and Inoue, Nakamasa and Saito, Kuniaki and Santo, Hiroaki and Okura, Fumio},
  booktitle = {CVPR},
  title = {BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment},
  year = {2026},
}
"""

_DESCRIPTION_TEMPLATE = (
    "BioVITA {arrow} retrieval over wild-species recordings. Each query is scored "
    "against its own official 100-way candidate pool: 100 candidate taxa (the correct "
    "one plus 99 distractors), where a taxon is represented by every {doc_modality} "
    "sample of that taxon in the benchmark index. Following the official evaluation "
    "script, a taxon scores the maximum similarity over its samples and the 100 taxa "
    "are ranked by that score, so the reported `taxon_top_k_accuracy` metrics are "
    "taxon-level rather than document-level. The per-query pool is carried through "
    "`top_ranked`, so no query is ever scored against the full corpus. Text is the "
    "bare taxon name of the evaluated level, as embedded by the official feature "
    "extractor. Only two of the paper's six per-direction scenarios are implemented -- "
    "`unseen_species` and `unseen_genus` -- because the released dataset ships no "
    "task files for the seen split and its unseen family files contain no rows."
)


@cache
def _download_csv(relative_path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            _PATH,
            f"{_GROUP}/{relative_path}",
            repo_type="dataset",
            revision=_REVISION,
        )
    )


@cache
def _download_media(modality: str) -> Path:
    """Download only the `unseen` media for one modality and return the repo root.

    Cached because the six directions and two subsets would otherwise re-resolve
    the same few thousand files against the hub on every split build.
    """
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            _PATH,
            repo_type="dataset",
            revision=_REVISION,
            allow_patterns=[f"{_GROUP}/{modality}/**"],
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

    metadata = pd.read_csv(_download_csv("metadata.csv"))
    files_by_species: dict[str, list[Path]] = defaultdict(list)
    for name, taxon in zip(metadata["file_name"], metadata["scientific_name"]):
        files_by_species[str(taxon)].append(root / _GROUP / str(name))

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
        if (root / _GROUP / relative).exists():
            if path != root / _GROUP / relative:
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
def _audio_fallbacks(paths: tuple[Path, ...]) -> tuple[dict[str, Any], ...] | None:
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
                int(_FALLBACK_SAMPLE_RATE * _FALLBACK_SECONDS), dtype="float32"
            )
            sampling_rate = _FALLBACK_SAMPLE_RATE
            logger.warning(
                "BioVITA: %s cannot be decoded at all and is scored as silence, "
                "matching the official evaluation script's zero-waveform fallback",
                path.name,
            )
        payloads.append(
            {"bytes": _wav_bytes(samples, int(sampling_rate)), "path": str(path)}
        )
    logger.warning(
        "BioVITA: %d/%d published audio clips are rejected by this environment's "
        "decoder and were re-decoded in memory via the official librosa fallback",
        sum(1 for ok in decodes if not ok),
        len(paths),
    )
    return tuple(payloads)


@cache
def _load_index(modality: str, level: str) -> tuple[Any, list[str], list[str]]:
    """One modality index as a HF dataset of `id` plus the payload column.

    Cached across tasks and subsets: the same audio and image indices back all
    six retrieval directions, and rebuilding them per split is pure overhead.
    """
    import pandas as pd
    from datasets import Audio, Dataset, Image

    index = pd.read_csv(_download_csv(f"benchmark/{modality}_index.csv"))
    ids = [str(i) for i in index["id"]]
    taxa = [str(t) for t in index[level]]

    if modality == "text":
        # The official feature extractor embeds the bare taxon name of the
        # evaluated level, so genus-level text repeats across the species of a
        # genus. Those duplicates stay separate documents because they are
        # separate ids in the official candidate groups.
        return Dataset.from_dict({"id": ids, "text": taxa}), ids, taxa

    root = _download_media(modality)
    if modality == "image":
        paths = [str(root / _GROUP / path) for path in index["file_path"]]
        column, feature = "image", Image()
    else:
        resolved = _resolve_audio_paths(
            [str(path) for path in index["file_path"]],
            [str(name) for name in index["species"]],
            root,
        )
        fallbacks = _audio_fallbacks(tuple(resolved))
        paths = (
            [str(path) for path in resolved] if fallbacks is None else list(fallbacks)
        )
        column, feature = "audio", Audio()

    data = Dataset.from_dict({"id": ids, column: paths}).cast_column(column, feature)
    return data, ids, taxa


class _BioVITARetrieval(AbsTaskRetrieval):
    """Shared loader for the six BioVITA cross-modal retrieval directions."""

    csv_name: ClassVar[str]
    query_modality: ClassVar[str]
    document_modality: ClassVar[str]

    # The official evaluation reports Top-1/Top-5 accuracy; its script also
    # computes Top-10.
    k_values = (1, 5, 10)
    # Candidate pools reach 1509 documents, so `_top_k` must cover the largest
    # pool for every candidate to be scored and no taxon group to be dropped.
    _top_k = 2048

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self._candidate_taxa: dict[str, dict[str, dict[str, list[str]]]] = {}
        self._correct_taxon: dict[str, dict[str, dict[str, str]]] = {}
        self._doc_taxon: dict[str, dict[str, dict[str, str]]] = {}

        self.dataset = {
            subset: {"test": self._build_split(subset, level)}
            for subset, level in _SUBSET_LEVELS.items()
        }
        self.data_loaded = True

    def _build_split(self, subset: str, level: str) -> RetrievalSplitData:  # noqa: PLR0914
        import pandas as pd

        corpus, corpus_ids, corpus_taxa = _load_index(self.document_modality, level)
        query_index, query_index_ids, _ = _load_index(self.query_modality, level)

        tasks = pd.read_csv(_download_csv(f"benchmark/{level}/{self.csv_name}"))
        query_ids = [str(query_id) for query_id in tasks["query_id"]]
        row_of_payload = {payload: row for row, payload in enumerate(query_index_ids)}
        queries = (
            query_index.select(
                [row_of_payload[str(payload)] for payload in tasks["query_payload_id"]]
            )
            .remove_columns("id")
            .add_column("id", query_ids)
        )

        relevant_docs: dict[str, dict[str, int]] = {}
        top_ranked: dict[str, list[str]] = {}
        candidate_taxa: dict[str, list[str]] = {}
        correct_taxon: dict[str, str] = {}

        for query_id, correct, taxa_json, ids_json in zip(
            query_ids,
            tasks["correct_taxon"],
            tasks["candidates_taxa"],
            tasks["candidates_target_ids"],
        ):
            taxa = [str(taxon) for taxon in json.loads(taxa_json)]
            groups = [[str(int(i)) for i in group] for group in json.loads(ids_json)]
            candidate_taxa[query_id] = taxa
            correct_taxon[query_id] = str(correct)
            top_ranked[query_id] = [doc_id for group in groups for doc_id in group]
            relevant_docs[query_id] = dict.fromkeys(groups[taxa.index(str(correct))], 1)

        self._candidate_taxa.setdefault(subset, {})["test"] = candidate_taxa
        self._correct_taxon.setdefault(subset, {})["test"] = correct_taxon
        self._doc_taxon.setdefault(subset, {})["test"] = dict(
            zip(corpus_ids, corpus_taxa)
        )

        return RetrievalSplitData(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            top_ranked=top_ranked,
        )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        """Official BioVITA scoring: rank the 100 candidate taxa by max-pooled similarity.

        BioVITA does not rank documents -- it ranks *taxa*. Each query comes with
        100 candidate taxa, and a taxon is represented by every sample of that
        taxon in the index (1 text, but up to 95 images or 28 clips). Following
        `eval_benchmark.py`, a taxon scores the **maximum** similarity over its
        own samples, the 100 taxa are sorted by that score, and Top@k asks
        whether the correct taxon is among the top k taxa. Max-pooling is what
        makes the taxon the unit of competition: a species with 15 images must
        not out-rank one with 3 simply by having more chances to appear in a
        document-level top-k list.

        MTEB's built-in retrieval metrics cannot express this, because they rank
        and count individual documents rather than groups of documents. Where a
        taxon owns several samples, the two readings come apart:

        * Recall@k divides by the number of relevant documents, so a query whose
          correct taxon holds several samples cannot reach 1.0 by retrieving one
          of them -- yet the official metric counts that as a hit.
        * Top@1 alone has a document-level twin: the top-ranked document's taxon
          is the top-ranked taxon, so precision@1 and hit_rate@1 agree with it.
        * For k > 1 there is no correspondence at all, because k taxa can span an
          arbitrary number of documents, so no document-level cut-off matches the
          official k-taxon cut-off.

        `taxon_top_1_accuracy` is therefore the `main_score`: it is the paper's
        headline metric (reported there as "Top-1 accuracy") and the only k where
        the official and document-level readings coincide. The `taxon_` prefix
        keeps it visibly distinct from the document-level `accuracy` below.

        Note for readers of the result files: the standard `accuracy`,
        `recall_at_k`, `ndcg_at_k` ... entries are still emitted by
        `make_score_dict` and are document-level; they are *not* the official
        BioVITA numbers and are not comparable with the paper.
        """
        candidate_taxa = self._candidate_taxa[hf_subset][hf_split]
        correct_taxon = self._correct_taxon[hf_subset][hf_split]
        doc_taxon = self._doc_taxon[hf_subset][hf_split]

        hits = dict.fromkeys(self.k_values, 0)
        total = 0
        for query_id, doc_scores in results.items():
            taxa = candidate_taxa[query_id]
            rank_of_taxon = {taxon: rank for rank, taxon in enumerate(taxa)}
            best: dict[str, float] = {}
            for doc_id, score in doc_scores.items():
                taxon = doc_taxon.get(doc_id)
                if taxon is None or taxon not in rank_of_taxon:
                    continue
                if taxon not in best or score > best[taxon]:
                    best[taxon] = score
            # Ties keep the candidate order of the official CSV, matching the
            # index order `torch.topk` falls back on in the reference script.
            ranked = sorted(
                taxa,
                key=lambda taxon: (
                    -best.get(taxon, float("-inf")),
                    rank_of_taxon[taxon],
                ),
            )
            total += 1
            correct = correct_taxon[query_id]
            for k in self.k_values:
                if correct in ranked[:k]:
                    hits[k] += 1

        return {
            f"taxon_top_{k}_accuracy": hits[k] / max(1, total) for k in self.k_values
        }


class BioVITAA2TRetrieval(_BioVITARetrieval):
    csv_name = "test_audio_to_text.csv"
    query_modality = "audio"
    document_modality = "text"

    metadata = TaskMetadata(
        name="BioVITAA2TRetrieval",
        description=_DESCRIPTION_TEMPLATE.format(
            arrow="audio-to-text", doc_modality="text"
        ),
        reference=_REFERENCE,
        dataset={"path": _PATH, "revision": _REVISION},
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAT2ARetrieval(_BioVITARetrieval):
    csv_name = "test_text_to_audio.csv"
    query_modality = "text"
    document_modality = "audio"

    metadata = TaskMetadata(
        name="BioVITAT2ARetrieval",
        description=_DESCRIPTION_TEMPLATE.format(
            arrow="text-to-audio", doc_modality="audio"
        ),
        reference=_REFERENCE,
        dataset={"path": _PATH, "revision": _REVISION},
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAA2IRetrieval(_BioVITARetrieval):
    csv_name = "test_audio_to_image.csv"
    query_modality = "audio"
    document_modality = "image"

    metadata = TaskMetadata(
        name="BioVITAA2IRetrieval",
        description=_DESCRIPTION_TEMPLATE.format(
            arrow="audio-to-image", doc_modality="image"
        ),
        reference=_REFERENCE,
        dataset={"path": _PATH, "revision": _REVISION},
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAI2ARetrieval(_BioVITARetrieval):
    csv_name = "test_image_to_audio.csv"
    query_modality = "image"
    document_modality = "audio"

    metadata = TaskMetadata(
        name="BioVITAI2ARetrieval",
        description=_DESCRIPTION_TEMPLATE.format(
            arrow="image-to-audio", doc_modality="audio"
        ),
        reference=_REFERENCE,
        dataset={"path": _PATH, "revision": _REVISION},
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Bioacoustics", "Nature", "Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAI2TRetrieval(_BioVITARetrieval):
    csv_name = "test_image_to_text.csv"
    query_modality = "image"
    document_modality = "text"

    metadata = TaskMetadata(
        name="BioVITAI2TRetrieval",
        description=_DESCRIPTION_TEMPLATE.format(
            arrow="image-to-text", doc_modality="text"
        ),
        reference=_REFERENCE,
        dataset={"path": _PATH, "revision": _REVISION},
        type="Any2AnyRetrieval",
        category="i2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Nature", "Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )


class BioVITAT2IRetrieval(_BioVITARetrieval):
    csv_name = "test_text_to_image.csv"
    query_modality = "text"
    document_modality = "image"

    metadata = TaskMetadata(
        name="BioVITAT2IRetrieval",
        description=_DESCRIPTION_TEMPLATE.format(
            arrow="text-to-image", doc_modality="image"
        ),
        reference=_REFERENCE,
        dataset={"path": _PATH, "revision": _REVISION},
        type="Any2AnyRetrieval",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={"unseen_species": ["eng-Latn"], "unseen_genus": ["eng-Latn"]},
        main_score="taxon_top_1_accuracy",
        date=("1960-01-01", "2026-05-15"),
        domains=["Nature", "Encyclopaedic"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )
