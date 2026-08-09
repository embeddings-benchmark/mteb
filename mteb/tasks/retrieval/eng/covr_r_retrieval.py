from pathlib import Path

from datasets import Video, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


def _video_id(value) -> str:
    """Return the flat filename stem used by the dataset's videos directory."""
    return str(value).rsplit("/", maxsplit=1)[-1]


class CoVRRVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CoVRRVT2VRetrieval",
        description=(
            "CoVR-R is a reasoning-aware benchmark for composed video retrieval. "
            "Given a reference video and a textual modification, the goal is to "
            "retrieve the correct target video that reflects the requested change "
            "and its implied visual consequences."
        ),
        reference="https://arxiv.org/abs/2603.20190",
        dataset={
            "path": "omkarthawakar/CoVR-R",
            "revision": "5e4543c680b19238bbb773e6757563c28d5666d8",
        },
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=("2026-03-20", "2026-03-20"),
        domains=["Web", "Activity"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{thawakar2026covrr,
  author = {Thawakar, Omkar and Demidov, Dmitry and Potlapalli, Vaishnav and Bogireddy, Sai Prasanna Teja Reddy and Gajjala, Viswanatha Reddy and Lasheen, Alaa Mostafa and Anwer, Rao Muhammad and Khan, Fahad Shahbaz},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  title = {CoVR-R: Reason-Aware Composed Video Retrieval},
  year = {2026},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load the release annotations and pair them with the flat video files."""
        if self.data_loaded:
            return

        grouped_annotations = load_dataset(
            self.metadata.dataset["path"],
            data_files="merged_webvid_ss2.json",
            revision=self.metadata.dataset["revision"],
            split="train",
            num_proc=num_proc,
        )

        annotations = []
        for group in grouped_annotations:
            for source_name, rows in group.items():
                if rows is not None:
                    annotations.extend((source_name, row) for row in rows)

        videos = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split="train",
            num_proc=num_proc,
        )
        video_paths = videos.cast_column("video", Video(decode=False))["video"]
        video_ids = [Path(video["path"]).stem for video in video_paths]
        video_index = {video_id: index for index, video_id in enumerate(video_ids)}

        missing_video_ids = {
            video_id
            for _, row in annotations
            for video_id in (
                _video_id(row["video_source"]),
                _video_id(row["video_target"]),
            )
            if video_id not in video_index
        }
        if missing_video_ids:
            missing = ", ".join(sorted(missing_video_ids)[:5])
            raise ValueError(f"CoVR-R is missing referenced videos: {missing}")

        query_ids = [f"{source_name}-{row['id']}" for source_name, row in annotations]
        source_ids = [_video_id(row["video_source"]) for _, row in annotations]
        target_ids = [_video_id(row["video_target"]) for _, row in annotations]

        queries = videos.select(
            [video_index[source_id] for source_id in source_ids]
        ).add_column("id", query_ids)
        queries = queries.add_column(
            "text", [row["modification_text"] for _, row in annotations]
        )
        corpus = videos.add_column("id", video_ids)
        qrels = {
            query_id: {target_id: 1}
            for query_id, target_id in zip(query_ids, target_ids, strict=True)
        }
        top_ranked = {
            query_id: [video_id for video_id in video_ids if video_id != source_id]
            for query_id, source_id in zip(query_ids, source_ids, strict=True)
        }

        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=qrels,
                    top_ranked=top_ranked,
                )
            }
        }
        self.data_loaded = True
