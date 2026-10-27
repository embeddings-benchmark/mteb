from __future__ import annotations

from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

# a corpus video that can't be decoded ("Invalid data found when processing input");
# the one query it is relevant to has other relevant videos
_BROKEN_VIDEO_IDS = {"20130923/Early_Start_20130923_0200_cc_segment/07.700k.mp4"}


def _drop_broken_videos(task: AbsTaskRetrieval, num_proc: int | None) -> None:
    for subset in task.dataset.values():
        for split in subset.values():
            split["corpus"] = split["corpus"].filter(
                lambda video_id: video_id not in _BROKEN_VIDEO_IDS,
                input_columns="id",
                num_proc=num_proc,
            )
            split["relevant_docs"] = {
                query_id: {
                    doc_id: score
                    for doc_id, score in docs.items()
                    if doc_id not in _BROKEN_VIDEO_IDS
                }
                for query_id, docs in split["relevant_docs"].items()
            }


class StanfordI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StanfordI2VRetrieval",
        description=(
            "Image-to-video+audio scene retrieval on the official Stanford I2V 600K "
            "release. It contains 229 image queries and 3,401 newscast video "
            "clips, preserving the full release's query and relevance manifests "
            "while using the recommended smaller distractor corpus. The source's "
            "2,260 temporal annotations are represented as 1,280 unique binary "
            "query-video relevance judgments, following its scene-retrieval scorer."
        ),
        reference="https://doi.org/10.1145/2713168.2713197",
        dataset={
            "path": "Cerru02/Stanford-I2V-600K",
            "revision": "30ba417f4ed92c7035abd552cdef1ebfb3542186",
        },
        type="Any2AnyRetrieval",
        category="i2va",
        modalities=["image", "video", "audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="map_at_100",
        date=("2012-10-01", "2013-09-30"),
        domains=["News"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{araujo2015stanford,
  author = {Araujo, Andr{\'e} and Chaves, Jason and Chen, David and Angst, Roland and Girod, Bernd},
  booktitle = {Proceedings of the 6th ACM Multimedia Systems Conference},
  pages = {237--242},
  title = {Stanford I2V: a news video dataset for query-by-image experiments},
  year = {2015},
}

@article{AraujoArxiv2016,
  author = {Araujo, A. and Chaves, J. and Lakshman, H. and Angst, R. and Girod, B.},
  journal = {arXiv preprint arXiv:1604.07939},
  title = {{Large-Scale Query-by-Image Video Retrieval Using Bloom Filters}},
  year = {2016},
}
""",
        prompt={
            "query": (
                "Retrieve news video+audio clips containing the visual event or "
                "scene shown in the image."
            )
        },
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _drop_broken_videos(self, num_proc)


class StanfordI2VVisualRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StanfordI2VVisualRetrieval",
        description=(
            "Visual-only image-to-video scene retrieval on the official Stanford I2V "
            "600K release. The original relevance labels were created by annotators "
            "marking video intervals where each query image appeared in the frames; "
            "audio was not used. It contains 229 image queries and 3,401 newscast "
            "video clips, preserving the full release's query and relevance manifests "
            "while using the recommended smaller distractor corpus. The source's "
            "2,260 temporal annotations are represented as 1,280 unique binary "
            "query-video relevance judgments, following its scene-retrieval scorer."
        ),
        reference="https://doi.org/10.1145/2713168.2713197",
        dataset={
            "path": "Cerru02/Stanford-I2V-600K",
            "revision": "30ba417f4ed92c7035abd552cdef1ebfb3542186",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="map_at_100",
        date=("2012-10-01", "2013-09-30"),
        domains=["News"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{araujo2015stanford,
  author = {Araujo, Andr{\'e} and Chaves, Jason and Chen, David and Angst, Roland and Girod, Bernd},
  booktitle = {Proceedings of the 6th ACM Multimedia Systems Conference},
  pages = {237--242},
  title = {Stanford I2V: a news video dataset for query-by-image experiments},
  year = {2015},
}

@article{AraujoArxiv2016,
  author = {Araujo, A. and Chaves, J. and Lakshman, H. and Angst, R. and Girod, B.},
  journal = {arXiv preprint arXiv:1604.07939},
  title = {{Large-Scale Query-by-Image Video Retrieval Using Bloom Filters}},
  year = {2016},
}
""",
        prompt={
            "query": (
                "Retrieve news video clips containing the visual event or scene shown "
                "in the image."
            )
        },
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        _drop_broken_videos(self, num_proc)
        for split in self.metadata.eval_splits:
            corpus = self.dataset["default"][split]["corpus"]
            if "audio" in corpus.column_names:
                self.dataset["default"][split]["corpus"] = corpus.remove_columns(
                    "audio"
                )
