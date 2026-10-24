#!/usr/bin/env python3
"""Measure how well MMEdit targets can be retrieved without the edit instruction.

This diagnostic intentionally ignores the text instruction. It embeds each source
and target WAV with a lightweight, order-insensitive log-mel fingerprint and ranks
targets by cosine similarity. The fingerprint is not intended as a competitive
audio model; it is a cheap lower-bound check for source-content leakage that runs
on CPU without downloading model weights.

Run a deterministic, stratified subset first:

  uv run python scripts/data/mmedit_retrieval/diagnose_source_only.py \
      --source-dir /path/to/MMEdit-TestSet --limit 512

Use ``--limit 0`` only after reviewing the subset result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from wave import open as wave_open

import numpy as np
from scipy.signal import resample_poly, spectrogram
from tqdm import tqdm

SOURCE_REPO_ID = "CocoBro/MMEdit-TestSet"
SOURCE_REVISION = "ae4f9a772180a2a3c77c2e865b398e7d6f60bcee"
EXPECTED_TRIPLETS = 3_317
TARGET_SAMPLE_RATE = 16_000
N_MELS = 64
_AUDIO_ID_RE = re.compile(r"^(?P<family>.+)_(?P<number>\d+)$")
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Retrieval metrics for binary qrels at a fixed cutoff of ten."""

    recall_at_1: float
    map_at_10: float
    ndcg_at_10: float


@dataclass
class DiagnosticResult:
    """Machine-readable result from the source-only diagnostic."""

    source_repo_id: str
    source_revision: str
    source_dir: str
    method: str
    seed: int
    requested_limit: int
    query_count: int
    corpus_count: int
    family_counts: dict[str, int]
    metrics: RetrievalMetrics
    duration_then_fingerprint_metrics: RetrievalMetrics
    random_expected: RetrievalMetrics
    recall_at_1_by_family: dict[str, float]
    top_false_positives: list[dict[str, Any]]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _external_source_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if _is_relative_to(resolved, _REPO_ROOT):
        raise SystemExit(
            f"--source-dir must be outside the Git repository ({_REPO_ROOT}): "
            f"{resolved}"
        )
    for required in ("content.jsonl", "raw", "target"):
        if not (resolved / required).exists():
            raise SystemExit(f"Missing required source path: {resolved / required}")
    return resolved


def _load_complete_ids(source_dir: Path) -> list[str]:
    metadata_ids: list[str] = []
    with (source_dir / "content.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            audio_id = row.get("audio_id")
            if not isinstance(audio_id, str) or not audio_id:
                raise SystemExit(f"Invalid audio_id on metadata line {line_number}")
            metadata_ids.append(audio_id)

    if len(metadata_ids) != len(set(metadata_ids)):
        raise SystemExit("content.jsonl contains duplicate audio_id values")
    complete_ids = [
        audio_id
        for audio_id in metadata_ids
        if (source_dir / "raw" / f"{audio_id}.wav").is_file()
        and (source_dir / "target" / f"{audio_id}.wav").is_file()
    ]
    if len(complete_ids) != EXPECTED_TRIPLETS:
        raise SystemExit(
            f"Expected {EXPECTED_TRIPLETS:,} complete triplets, "
            f"found {len(complete_ids):,}; run create_data.py first"
        )
    return sorted(complete_ids)


def _family(audio_id: str) -> str:
    match = _AUDIO_ID_RE.fullmatch(audio_id)
    if match is None:
        raise ValueError(f"Invalid MMEdit audio_id: {audio_id!r}")
    return match.group("family")


def _stable_order(audio_ids: list[str], seed: int) -> list[str]:
    return sorted(
        audio_ids,
        key=lambda audio_id: hashlib.sha256(f"{seed}:{audio_id}".encode()).digest(),
    )


def _stratified_subset(audio_ids: list[str], limit: int, seed: int) -> list[str]:
    """Select a proportional subset while retaining every edit family."""
    if limit == 0 or limit >= len(audio_ids):
        return sorted(audio_ids)

    by_family: dict[str, list[str]] = defaultdict(list)
    for audio_id in audio_ids:
        by_family[_family(audio_id)].append(audio_id)
    if limit < len(by_family):
        raise ValueError(
            f"--limit must be 0 or at least the {len(by_family)} edit families"
        )

    family_sizes = {family: len(ids) for family, ids in by_family.items()}
    quotas = {family: 1 for family in by_family}
    remaining = limit - len(quotas)
    total_after_minimum = sum(size - 1 for size in family_sizes.values())
    fractional: list[tuple[float, str]] = []
    assigned = 0
    for family, size in family_sizes.items():
        exact = remaining * (size - 1) / total_after_minimum
        addition = math.floor(exact)
        quotas[family] += addition
        assigned += addition
        fractional.append((exact - addition, family))
    for _, family in sorted(fractional, key=lambda item: (-item[0], item[1]))[
        : remaining - assigned
    ]:
        quotas[family] += 1

    selected = []
    for family in sorted(by_family):
        selected.extend(_stable_order(by_family[family], seed)[: quotas[family]])
    return sorted(selected)


def _read_pcm_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave_open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        compression = wav.getcomptype()
        payload = wav.readframes(frames)

    if compression != "NONE":
        raise ValueError(f"Unsupported compressed WAV {path}: {compression}")
    dtype_and_scale = {
        1: (np.dtype("u1"), 128.0),
        2: (np.dtype("<i2"), 32768.0),
        4: (np.dtype("<i4"), 2147483648.0),
    }
    if sample_width not in dtype_and_scale:
        raise ValueError(f"Unsupported {sample_width}-byte PCM WAV: {path}")
    dtype, scale = dtype_and_scale[sample_width]
    audio = np.frombuffer(payload, dtype=dtype).astype(np.float32)
    if sample_width == 1:
        audio -= 128.0
    audio /= scale
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    return audio, TARGET_SAMPLE_RATE


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    frequencies = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(50.0), _hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ),
        )
    return filters


_MEL_FILTERS = _mel_filterbank(TARGET_SAMPLE_RATE, n_fft=512, n_mels=N_MELS)


def _spectral_fingerprint(path: Path) -> np.ndarray:
    audio, sample_rate = _read_pcm_mono(path)
    if len(audio) == 0:
        raise ValueError(f"Empty WAV: {path}")

    # Remove gain as an easy cue so that the diagnostic primarily measures
    # shared acoustic content, including for MMEdit's loudness operations.
    audio = audio - np.mean(audio)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms > 1e-8:
        audio = audio / rms

    _, _, power = spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=400,
        noverlap=240,
        nfft=512,
        detrend=False,
        scaling="spectrum",
        mode="psd",
    )
    mel_power = _MEL_FILTERS @ power
    mel_power /= np.maximum(mel_power.sum(axis=0, keepdims=True), 1e-12)
    log_mel = np.log(np.maximum(mel_power, 1e-10))
    temporal_change = np.abs(np.diff(log_mel, axis=1)).mean(axis=1)
    fingerprint = np.concatenate(
        [
            log_mel.mean(axis=1),
            log_mel.std(axis=1),
            np.quantile(log_mel, 0.25, axis=1),
            np.quantile(log_mel, 0.50, axis=1),
            np.quantile(log_mel, 0.75, axis=1),
            temporal_change,
        ]
    )
    return fingerprint.astype(np.float32)


def _embed(paths: list[Path], *, description: str) -> np.ndarray:
    embeddings = [
        _spectral_fingerprint(path)
        for path in tqdm(paths, desc=description, unit="file")
    ]
    return np.vstack(embeddings)


def _normalize_embeddings(
    source_embeddings: np.ndarray, target_embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit unsupervised feature scaling on sources, then L2-normalize both sets."""
    center = source_embeddings.mean(axis=0, keepdims=True)
    scale = source_embeddings.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    source_embeddings = (source_embeddings - center) / scale
    target_embeddings = (target_embeddings - center) / scale
    source_embeddings /= np.maximum(
        np.linalg.norm(source_embeddings, axis=1, keepdims=True), 1e-12
    )
    target_embeddings /= np.maximum(
        np.linalg.norm(target_embeddings, axis=1, keepdims=True), 1e-12
    )
    return source_embeddings, target_embeddings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_seconds(path: Path) -> float:
    with wave_open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def _relevant_indices(target_paths: list[Path]) -> list[set[int]]:
    """Treat byte-identical target candidates as equally relevant."""
    hashes = [_sha256(path) for path in target_paths]
    by_hash: dict[str, set[int]] = defaultdict(set)
    for index, digest in enumerate(hashes):
        by_hash[digest].add(index)
    return [by_hash[digest] for digest in hashes]


def _metrics_from_order(
    order: np.ndarray, relevant: list[set[int]], *, cutoff: int = 10
) -> tuple[RetrievalMetrics, np.ndarray]:
    recalls = []
    average_precisions = []
    ndcgs = []
    first_relevant_ranks = []
    for ranking, positives in zip(order, relevant):
        first_rank = next(
            rank
            for rank, candidate in enumerate(ranking, start=1)
            if candidate in positives
        )
        first_relevant_ranks.append(first_rank)
        recalls.append(float(first_rank == 1))

        hits = 0
        precision_sum = 0.0
        dcg = 0.0
        for rank, candidate in enumerate(ranking[:cutoff], start=1):
            if candidate not in positives:
                continue
            hits += 1
            precision_sum += hits / rank
            dcg += 1.0 / math.log2(rank + 1)
        average_precisions.append(precision_sum / min(len(positives), cutoff))
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(positives), cutoff) + 1)
        )
        ndcgs.append(dcg / ideal_dcg)

    return (
        RetrievalMetrics(
            recall_at_1=float(np.mean(recalls)),
            map_at_10=float(np.mean(average_precisions)),
            ndcg_at_10=float(np.mean(ndcgs)),
        ),
        np.asarray(first_relevant_ranks),
    )


def _random_expected(relevant: list[set[int]], corpus_size: int) -> RetrievalMetrics:
    """Compute exact expected metrics for a uniformly random candidate ordering."""
    recall_values = []
    ap_values = []
    ndcg_values = []
    cutoff = min(10, corpus_size)
    for positives in relevant:
        relevant_count = len(positives)
        recall_values.append(relevant_count / corpus_size)

        # At rank r, a relevant item occurs with probability R/N. Conditional on
        # that hit, expected relevant hits through r is 1 + (r-1)(R-1)/(N-1).
        expected_precision_sum = 0.0
        expected_dcg = 0.0
        for rank in range(1, cutoff + 1):
            hit_probability = relevant_count / corpus_size
            earlier_hits = (
                (rank - 1) * (relevant_count - 1) / (corpus_size - 1)
                if corpus_size > 1
                else 0.0
            )
            expected_precision_sum += hit_probability * (1 + earlier_hits) / rank
            expected_dcg += hit_probability / math.log2(rank + 1)
        ap_values.append(expected_precision_sum / min(relevant_count, cutoff))
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(relevant_count, cutoff) + 1)
        )
        ndcg_values.append(expected_dcg / ideal_dcg)

    return RetrievalMetrics(
        recall_at_1=float(np.mean(recall_values)),
        map_at_10=float(np.mean(ap_values)),
        ndcg_at_10=float(np.mean(ndcg_values)),
    )


def diagnose(source_dir: Path, *, limit: int, seed: int) -> DiagnosticResult:
    all_ids = _load_complete_ids(source_dir)
    selected_ids = _stratified_subset(all_ids, limit=limit, seed=seed)
    source_paths = [source_dir / "raw" / f"{audio_id}.wav" for audio_id in selected_ids]
    target_paths = [
        source_dir / "target" / f"{audio_id}.wav" for audio_id in selected_ids
    ]

    source_embeddings = _embed(source_paths, description="Embed source audio")
    target_embeddings = _embed(target_paths, description="Embed target audio")
    source_embeddings, target_embeddings = _normalize_embeddings(
        source_embeddings, target_embeddings
    )
    scores = source_embeddings @ target_embeddings.T
    order = np.argsort(-scores, axis=1, kind="stable")
    relevant = _relevant_indices(target_paths)
    metrics, first_relevant_ranks = _metrics_from_order(order, relevant)

    # MMEdit's source and target have exactly matching lengths, while 1,394 pairs
    # violate the advertised ten-second duration. Measure that additional leakage
    # explicitly: duration distance is the primary key and the spectral score only
    # resolves clips with the same duration.
    source_durations = np.asarray([_duration_seconds(path) for path in source_paths])
    target_durations = np.asarray([_duration_seconds(path) for path in target_paths])
    duration_distances = np.abs(
        source_durations[:, np.newaxis] - target_durations[np.newaxis, :]
    )
    duration_then_fingerprint_order = np.vstack(
        [
            np.lexsort((-scores[index], duration_distances[index]))
            for index in range(len(selected_ids))
        ]
    )
    duration_then_fingerprint_metrics, _ = _metrics_from_order(
        duration_then_fingerprint_order, relevant
    )

    family_counts = Counter(_family(audio_id) for audio_id in selected_ids)
    recall_by_family = {}
    for family in sorted(family_counts):
        family_hits = [
            first_relevant_ranks[index] == 1
            for index, audio_id in enumerate(selected_ids)
            if _family(audio_id) == family
        ]
        recall_by_family[family] = float(np.mean(family_hits))

    false_positives = []
    for query_index, rank in enumerate(first_relevant_ranks):
        if rank == 1:
            continue
        predicted_index = int(order[query_index, 0])
        false_positives.append(
            {
                "query_id": selected_ids[query_index],
                "query_family": _family(selected_ids[query_index]),
                "predicted_target_id": selected_ids[predicted_index],
                "predicted_target_family": _family(selected_ids[predicted_index]),
                "positive_rank": int(rank),
                "predicted_score": float(scores[query_index, predicted_index]),
                "positive_score": float(scores[query_index, query_index]),
            }
        )
    false_positives.sort(
        key=lambda row: (row["positive_rank"], row["query_id"]), reverse=True
    )

    return DiagnosticResult(
        source_repo_id=SOURCE_REPO_ID,
        source_revision=SOURCE_REVISION,
        source_dir=str(source_dir),
        method=(
            "source-only order-invariant 64-bin log-mel distribution fingerprint; "
            "cosine similarity"
        ),
        seed=seed,
        requested_limit=limit,
        query_count=len(selected_ids),
        corpus_count=len(selected_ids),
        family_counts=dict(sorted(family_counts.items())),
        metrics=metrics,
        duration_then_fingerprint_metrics=duration_then_fingerprint_metrics,
        random_expected=_random_expected(relevant, len(selected_ids)),
        recall_at_1_by_family=recall_by_family,
        top_false_positives=false_positives[:20],
    )


def _print_metrics(label: str, metrics: RetrievalMetrics) -> None:
    print(
        f"  {label:<20} "
        f"R@1={metrics.recall_at_1:.4f}  "
        f"mAP@10={metrics.map_at_10:.4f}  "
        f"nDCG@10={metrics.ndcg_at_10:.4f}"
    )


def _print_result(result: DiagnosticResult) -> None:
    print("\nMMEdit source-only retrieval diagnostic")
    print(f"  Revision:            {result.source_revision}")
    print(f"  Queries/corpus:      {result.query_count:,}/{result.corpus_count:,}")
    print(f"  Deterministic seed:  {result.seed}")
    _print_metrics("Spectral fingerprint", result.metrics)
    _print_metrics("Duration + spectral", result.duration_then_fingerprint_metrics)
    _print_metrics("Random expectation", result.random_expected)
    family_scores = ", ".join(
        f"{family}={score:.2f}"
        for family, score in result.recall_at_1_by_family.items()
    )
    print(f"  R@1 by family:       {family_scores}")
    if result.top_false_positives:
        print("  Highest-rank misses:")
        for row in result.top_false_positives[:8]:
            print(
                f"    {row['query_id']} -> {row['predicted_target_id']} "
                f"(positive rank {row['positive_rank']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=512,
        help="Number of stratified triplets to evaluate; 0 uses all 3,317.",
    )
    parser.add_argument(
        "--seed", type=int, default=5140, help="Deterministic subset selection seed."
    )
    parser.add_argument(
        "--json-report", type=Path, help="Optionally write the result as JSON."
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")
    source_dir = _external_source_dir(args.source_dir)
    try:
        result = diagnose(source_dir, limit=args.limit, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    _print_result(result)

    if args.json_report is not None:
        report_path = args.json_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {report_path}")


if __name__ == "__main__":
    main()
