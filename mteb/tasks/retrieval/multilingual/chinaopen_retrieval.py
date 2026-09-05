from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "zho-Hans": ["zho-Hans"],
    "eng-Latn": ["eng-Latn"],
}

_CHINAOPEN_BIBTEX = r"""
@inproceedings{chen2023chinaopen,
  author = {Chen, Aozhu and Wang, Ziyuan and Dong, Chengbo and Tian, Kaibin and Zhao, Ruixiang and Liang, Xun and Kang, Zhanhui and Li, Xirong},
  booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
  doi = {10.1145/3581783.3612156},
  title = {ChinaOpen: A Dataset for Open-world Multimodal Learning},
  year = {2023},
}
"""

_CHINAOPEN_DESCRIPTION_TAIL = (
    "Built from the manually annotated ChinaOpen-1k test set (1,092 Bilibili "
    "videos). The Chinese captions are the native annotation, written by human "
    "annotators watching the video, and the English captions are translations "
    "of them, so both language subsets describe the same videos and differ only "
    "in language. Uploader-written video titles are not used. Queries are "
    "deduplicated by caption text and every video carrying a caption is marked "
    "relevant, so the few captions shared by more than one video are "
    "multi-positive rather than incorrectly scored."
)


class ChinaOpenT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChinaOpenT2VRetrieval",
        description=(
            "Multilingual text-to-video retrieval over Chinese web video: given "
            "a caption in Chinese or English, retrieve the video it describes. "
            + _CHINAOPEN_DESCRIPTION_TAIL
        ),
        reference="https://ruc-aimc-lab.github.io/ChinaOpen/",
        dataset={
            "path": "shriyasudhakar/ChinaOpen1k-T2V",
            "revision": "3b7bc48fa4c6d88b50f336937bd0b6913e3fe5e7",
        },
        type="Any2AnyMultilingualRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        modalities=["text", "video"],
        date=("2023-01-01", "2023-12-31"),
        domains=["Web", "Entertainment"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_CHINAOPEN_BIBTEX,
        prompt={"query": "Find the video that matches the given caption."},
    )


class ChinaOpenV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChinaOpenV2TRetrieval",
        description=(
            "Multilingual video-to-text retrieval over Chinese web video: given "
            "a video, retrieve the caption describing it from a corpus of "
            "Chinese or English captions. " + _CHINAOPEN_DESCRIPTION_TAIL
        ),
        reference="https://ruc-aimc-lab.github.io/ChinaOpen/",
        dataset={
            "path": "shriyasudhakar/ChinaOpen1k-V2T",
            "revision": "6e3af06f3bd28241a41d46e4b93c14420e70ce2a",
        },
        type="Any2AnyMultilingualRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2023-01-01", "2023-12-31"),
        domains=["Web", "Entertainment"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_CHINAOPEN_BIBTEX,
        prompt={"query": "Find the caption that describes the following video."},
    )
