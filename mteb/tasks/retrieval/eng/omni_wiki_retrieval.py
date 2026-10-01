from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@article{ma2024unifying,
  author = {Ma, Xueguang and Lin, Sheng-Chieh and Li, Minghan and Chen, Wenhu and Lin, Jimmy},
  journal = {arXiv preprint arXiv:2406.11251},
  title = {Unifying Multimodal Retrieval via Document Screenshot Embedding},
  year = {2024},
}
"""


class OmniWikiV2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmniWikiV2IRetrieval",
        description=(
            "Retrieve the Wikipedia page screenshot on which a given video is embedded. Each query "
            "is a video embedded on an English Wikipedia page (drawn from Tevatron Wiki-SS-NQ train "
            "positives, kept only if 10-180s and decodable by torchcodec), and the gold documents "
            "are the page-screenshot images of every corpus page the video appears on. The corpus "
            "is those gold pages plus the BM25 hard negatives of the questions whose answer page "
            "carries a query video."
        ),
        reference="https://arxiv.org/abs/2406.11251",
        dataset={
            "path": "whybe-choi/OmniWikiRetrieval",
            "revision": "10dc41f341a15352ae3639f4779afd420ed86c86",
        },
        type="Any2AnyRetrieval",
        category="v2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "image"],
        date=("2001-01-15", "2026-09-08"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the Wikipedia page screenshot relevant to this video."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # keep only the image corpus columns; the unused text is never loaded/decoded
        for subset in self.dataset.values():
            for split_data in subset.values():
                corpus = split_data["corpus"]
                keep = [c for c in ("id", "image") if c in corpus.column_names]
                split_data["corpus"] = corpus.select_columns(keep)


class OmniWikiV2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OmniWikiV2TRetrieval",
        description=(
            "Retrieve the text of the Wikipedia page on which a given video is embedded. Each query "
            "is a video embedded on an English Wikipedia page (drawn from Tevatron Wiki-SS-NQ train "
            "positives, kept only if 10-180s and decodable by torchcodec), and the gold documents "
            "are the page texts (title prepended) of every corpus page the video appears on. The "
            "corpus is those gold pages plus the BM25 hard negatives of the questions whose answer "
            "page carries a query video."
        ),
        reference="https://arxiv.org/abs/2406.11251",
        dataset={
            "path": "whybe-choi/OmniWikiRetrieval",
            "revision": "10dc41f341a15352ae3639f4779afd420ed86c86",
        },
        type="Any2AnyRetrieval",
        category="v2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        modalities=["video", "text"],
        date=("2001-01-15", "2026-09-08"),
        domains=["Encyclopaedic", "Web"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the Wikipedia page text relevant to this video."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # keep only the text corpus columns; the unused image is never loaded/decoded
        for subset in self.dataset.values():
            for split_data in subset.values():
                corpus = split_data["corpus"]
                keep = [c for c in ("id", "text") if c in corpus.column_names]
                split_data["corpus"] = corpus.select_columns(keep)
