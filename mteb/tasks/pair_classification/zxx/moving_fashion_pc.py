from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, ClassVar, TypedDict

from datasets import Dataset, concatenate_datasets, load_dataset

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_NEGATIVE_SAMPLING_SEED = 42
_SOURCE_BY_IMAGE_SUFFIX = {".png": "hard", ".jpg": "regular"}


class PairManifest(TypedDict):
    video_id: list[str]
    image_id: list[str]
    label: list[int]
    source_subset: list[str]


def _stable_key(seed: int, *parts: object) -> bytes:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode()).digest()


def _source_group(image_id: str) -> str:
    suffix = PurePosixPath(image_id).suffix.lower()
    try:
        return _SOURCE_BY_IMAGE_SUFFIX[suffix]
    except KeyError as error:
        raise ValueError(
            f"Cannot determine the MovingFashion source group for {image_id!r}"
        ) from error


def _balanced_candidate_slots(
    candidates: Sequence[str], positive_images: Sequence[str], count: int, seed: int
) -> list[str]:
    """Allocate negative-image slots while balancing total media frequency."""
    if not candidates:
        raise ValueError("Cannot sample negatives without candidate images")

    positive_usage = Counter(positive_images)
    negative_usage: Counter[str] = Counter()
    candidate_order = sorted(candidates, key=lambda item: _stable_key(seed, item))
    candidate_rank = {candidate: rank for rank, candidate in enumerate(candidate_order)}
    slots = []
    for _ in range(count):
        candidate = min(
            candidate_order,
            key=lambda item: (
                positive_usage[item] + negative_usage[item],
                negative_usage[item],
                candidate_rank[item],
            ),
        )
        negative_usage[candidate] += 1
        slots.append(candidate)

    return sorted(
        slots,
        key=lambda item: _stable_key(
            seed, "slot", item, negative_usage[item], positive_usage[item]
        ),
    )


def _assign_negative_images(
    source: str,
    positives: Sequence[tuple[str, str]],
    candidates: Sequence[str],
    positive_images_by_video: Mapping[str, set[str]],
    seed: int,
) -> dict[tuple[str, str], str]:
    ordered_positives = sorted(
        positives,
        key=lambda row: _stable_key(seed, source, "positive", *row),
    )
    slots = _balanced_candidate_slots(
        candidates,
        [image for _, image in ordered_positives],
        len(ordered_positives),
        seed,
    )
    for offset_delta in range(len(slots)):
        offset = (seed + offset_delta) % len(slots)
        proposed = [
            slots[(row_index + offset) % len(slots)]
            for row_index in range(len(ordered_positives))
        ]
        if all(
            candidate not in positive_images_by_video[video_id]
            for (video_id, _), candidate in zip(
                ordered_positives, proposed, strict=True
            )
        ):
            return dict(zip(ordered_positives, proposed, strict=True))
    raise RuntimeError(f"Could not construct collision-free {source} negative pairs")


def build_balanced_pair_manifest(
    video_ids: Sequence[str],
    image_ids: Sequence[str],
    qrel_video_ids: Sequence[str],
    qrel_image_ids: Sequence[str],
    *,
    seed: int = _NEGATIVE_SAMPLING_SEED,
) -> PairManifest:
    """Create balanced positive and negative MovingFashion video-image pairs.

    Every official qrel is a positive. Each positive receives one negative image
    from the same official source subset. Candidate use is balanced before a
    deterministic cyclic assignment avoids every known positive for the video,
    including the source's genuine multi-positive videos.
    """
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("MovingFashion query IDs must be unique")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("MovingFashion corpus IDs must be unique")
    if len(qrel_video_ids) != len(qrel_image_ids):
        raise ValueError("MovingFashion qrel columns have different lengths")

    available_videos = set(video_ids)
    available_images = set(image_ids)
    positives = list(zip(qrel_video_ids, qrel_image_ids, strict=True))
    if len(positives) != len(set(positives)):
        raise ValueError("MovingFashion qrels contain duplicate pairs")
    if missing := {video for video, _ in positives} - available_videos:
        raise ValueError(f"Qrels reference missing video IDs: {sorted(missing)}")
    if missing := {image for _, image in positives} - available_images:
        raise ValueError(f"Qrels reference missing image IDs: {sorted(missing)}")

    positive_images_by_video: dict[str, set[str]] = defaultdict(set)
    positive_rows_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    candidates_by_source: dict[str, list[str]] = defaultdict(list)
    for video_id, image_id in positives:
        positive_images_by_video[video_id].add(image_id)
        positive_rows_by_source[_source_group(image_id)].append((video_id, image_id))
    for image_id in image_ids:
        candidates_by_source[_source_group(image_id)].append(image_id)

    negative_by_positive: dict[tuple[str, str], str] = {}
    for source, source_positives in sorted(positive_rows_by_source.items()):
        negative_by_positive.update(
            _assign_negative_images(
                source,
                source_positives,
                candidates_by_source[source],
                positive_images_by_video,
                seed,
            )
        )

    rows: list[tuple[str, str, int, str]] = []
    for positive in positives:
        video_id, image_id = positive
        source = _source_group(image_id)
        rows.append((video_id, image_id, 1, source))
        rows.append((video_id, negative_by_positive[positive], 0, source))
    rows.sort(key=lambda row: _stable_key(seed, "row", *row))

    return {
        "video_id": [row[0] for row in rows],
        "image_id": [row[1] for row in rows],
        "label": [row[2] for row in rows],
        "source_subset": [row[3] for row in rows],
    }


class MovingFashionV2IPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="MovingFashionV2IPairClassification",
        description=(
            "Video-to-shop-image pair classification derived from the official "
            "MovingFashion test associations. Its 1,341 available human-annotated "
            "matches are balanced with 1,341 deterministic mismatches sampled "
            "within the same official hard or regular source subset. Negative "
            "assignment excludes every true image for each video and balances "
            "image frequency across labels. Media is shared with the corresponding "
            "MovingFashion retrieval task rather than published a second time."
        ),
        reference="https://arxiv.org/abs/2110.02627",
        dataset={
            "path": "pranitchawla/MovingFashion",
            "revision": "29c9813e2826ef2f4398455528881ab3e181311b",
        },
        type="VideoPairClassification",
        category="v2i",
        modalities=["video", "image"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="max_ap",
        date=("2021-10-06", "2022-01-08"),
        domains=["E-commerce", "Social"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="multiple",
        adapted_from=["MovingFashionV2IRetrieval"],
        bibtex_citation=r"""
@misc{godi2021movingfashion,
  archiveprefix = {arXiv},
  author = {Godi, Marco and Joppi, Christian and Skenderi, Geri and Cristani, Marco},
  eprint = {2110.02627},
  primaryclass = {cs.CV},
  title = {MovingFashion: a Benchmark for the Video-to-Shop Challenge},
  year = {2021},
}
""",
        prompt=(
            "Represent this media item for matching clothing across social videos "
            "and shop images."
        ),
        is_beta=True,
    )

    input1_column_name: ClassVar[Mapping[str, str]] = {"video": "video"}
    input2_column_name: ClassVar[Mapping[str, str]] = {"image": "image"}
    input1_id_column_name = "video_id"
    input2_id_column_name = "image_id"
    input1_prompt_type = PromptType.query
    input2_prompt_type = PromptType.document
    label_column_name = "label"

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        queries = load_dataset(
            path, "queries", revision=revision, split="test", num_proc=num_proc
        )
        corpus = load_dataset(
            path, "corpus", revision=revision, split="test", num_proc=num_proc
        )
        qrels = load_dataset(
            path, "qrels", revision=revision, split="test", num_proc=num_proc
        )

        manifest = Dataset.from_dict(
            build_balanced_pair_manifest(
                queries["id"],
                corpus["id"],
                qrels["query-id"],
                qrels["corpus-id"],
            )
        )
        query_index = {item_id: index for index, item_id in enumerate(queries["id"])}
        corpus_index = {item_id: index for index, item_id in enumerate(corpus["id"])}
        videos = queries.select(
            [query_index[item_id] for item_id in manifest["video_id"]]
        ).select_columns(["video"])
        images = corpus.select(
            [corpus_index[item_id] for item_id in manifest["image_id"]]
        ).select_columns(["image"])

        self.dataset = {
            "test": concatenate_datasets([manifest, videos, images], axis=1)
        }
        self.data_loaded = True
