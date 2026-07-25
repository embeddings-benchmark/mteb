from __future__ import annotations

import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, Features, Value, Video, load_dataset

from mteb.abstasks import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

_VIDEOS_PER_CATEGORY = 8
_CACHE_DIR = Path(tempfile.gettempdir()) / "mteb_finevideo_clustering_cache"
_MAX_STREAM_ATTEMPTS = 5
_MAX_LOAD_SECONDS = 900
_FEATURES = Features({"video": Video(), "label": Value("string")})


class FineVideoClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FineVideoClustering",
        description=(
            "Clustering of YouTube videos into fine-grained content categories "
            "from FineVideo, a Creative Commons video dataset derived by "
            "filtering 1.9M videos from YouTube-Commons down to ~44K fully "
            "annotated videos. Sampled up to 8 videos per fine-grained "
            "category (content_fine_category, ~122 categories) directly from "
            "the source dataset at load time; no repackaged copy is "
            "distributed. Any use of this data must abide by the terms of the "
            "original Creative Commons licenses of the individual videos, per "
            "the FineVideo Terms of Use. Because a handful of categories are "
            "rare in the source stream and the upstream host can be slow, "
            "load_data() is time-boxed to roughly 15 minutes: it ships "
            "whichever subset of categories it managed to fill by then "
            "rather than guaranteeing full coverage of every category."
        ),
        reference="https://huggingface.co/blog/fine-video",
        dataset={
            "path": "HuggingFaceFV/finevideo",
            "revision": "84c74091e1c6ee7a5dffabfafb5c9033e4718883",
        },
        type="VideoClustering",
        category="v2c",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-09-01", "2024-09-19"),
        domains=["Web", "Scene"],
        task_subtypes=["Activity recognition"],
        license="multiple",
        annotations_creators="derived",
        dialect=[],
        modalities=["video"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@misc{farre2024finevideo,
  author = {Farr\'{e}, Miquel and Marafioti, Andres and Tunstall, Lewis and Von Werra, Leandro and Cuenca, Pedro and Wolf, Thomas},
  title = {FineVideo},
  url = {https://huggingface.co/datasets/HuggingFaceFV/finevideo},
  year = {2024},
}
""",
    )
    max_fraction_of_documents_to_embed = None
    input_column_name: str = "video"
    label_column_name: str = "label"

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        _CACHE_DIR.mkdir(exist_ok=True, parents=True)

        counts: dict[str, int] = defaultdict(int)
        records: list[dict] = []
        rows_scanned = 0
        rows_since_last_add = 0
        rows_to_skip = 0
        load_start = time.time()

        def _time_budget_exceeded() -> bool:
            return time.time() - load_start > _MAX_LOAD_SECONDS

        for attempt in range(1, _MAX_STREAM_ATTEMPTS + 1):
            if _time_budget_exceeded():
                logger.warning(
                    "FineVideoClustering: %ds time budget reached before "
                    "attempt %d; shipping the %d-video subset collected so far.",
                    _MAX_LOAD_SECONDS,
                    attempt,
                    len(records),
                )
                break
            try:
                ds = load_dataset(
                    self.metadata.dataset["path"],
                    revision=self.metadata.dataset["revision"],
                    split="train",
                    streaming=True,
                )
                for i, row in enumerate(ds):
                    if i < rows_to_skip:
                        continue
                    rows_scanned += 1
                    # Bail out mid-shard once the time budget is hit, rather
                    # than only checking between streaming attempts, since a
                    # single flaky shard can otherwise stall for a long time.
                    if rows_scanned % 50 == 0 and _time_budget_exceeded():
                        self.dataset = {
                            "test": Dataset.from_list(records, features=_FEATURES)
                        }
                        self.data_loaded = True
                        return
                    label = row["json"]["content_fine_category"]
                    if counts[label] >= _VIDEOS_PER_CATEGORY:
                        rows_since_last_add += 1
                        # Stop once quotas have clearly stopped filling and
                        # we've scanned a reasonable fraction of the
                        # dataset, so we don't stream through all ~44K rows
                        # just to skip most of them.
                        if rows_since_last_add >= 1500 and rows_scanned >= 4000:
                            self.dataset = {
                                "test": Dataset.from_list(records, features=_FEATURES)
                            }
                            self.data_loaded = True
                            return
                        continue
                    rows_since_last_add = 0
                    video_id = row["json"]["original_video_filename"]
                    video_path = _CACHE_DIR / f"{video_id}.mp4"
                    if not video_path.exists():
                        video_path.write_bytes(row["mp4"])
                    records.append(
                        {
                            "video": {"path": str(video_path), "bytes": None},
                            "label": label,
                        }
                    )
                    counts[label] += 1
                # Stream exhausted naturally before quotas/early-exit hit.
                break
            except Exception:
                rows_to_skip = rows_scanned
                logger.warning(
                    "FineVideoClustering: stream error on attempt %d/%d after "
                    "%d rows scanned, retrying with a fresh stream.",
                    attempt,
                    _MAX_STREAM_ATTEMPTS,
                    rows_scanned,
                )
                if attempt == _MAX_STREAM_ATTEMPTS:
                    raise
                time.sleep(5)

        self.dataset = {"test": Dataset.from_list(records, features=_FEATURES)}
        self.data_loaded = True
