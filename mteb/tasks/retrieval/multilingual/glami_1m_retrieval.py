from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GLAMI1MT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GLAMI1MT2IRetrieval",
        description=(
            "Retrieve fashion product images from multilingual product names and "
            "descriptions in GLAMI-1M."
        ),
        reference="https://arxiv.org/abs/2211.14451",
        dataset={
            "path": "artist/glami-1m-t2i-mteb",
            "revision": "20969505f3edaa4f2b239155ec16a4c986b1b195",
        },
        type="Any2AnyMultilingualRetrieval",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs={
            "bg": ["bul-Cyrl"],
            "cs": ["ces-Latn"],
            "el": ["ell-Grek"],
            "es": ["spa-Latn"],
            "et": ["est-Latn"],
            "hr": ["hrv-Latn"],
            "hu": ["hun-Latn"],
            "lt": ["lit-Latn"],
            "lv": ["lav-Latn"],
            "ro": ["ron-Latn"],
            "sk": ["slk-Latn"],
            "sl": ["slv-Latn"],
            "tr": ["tur-Latn"],
        },
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["E-commerce", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{Kosar_2022_BMVC,
  author = {Vaclav Kosar and Antonín Hoskovec and Milan Šulc and Radek Bartyzal},
  booktitle = {33rd British Machine Vision Conference 2022, BMVC 2022, London, UK, November 21-24, 2022},
  publisher = {BMVA Press},
  title = {{GLAMI-1M}: A Multilingual Image-Text Fashion Dataset},
  url = {https://arxiv.org/abs/2211.14451},
  year = {2022},
}
""",
        prompt={"query": "Find the fashion product image matching this description."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.hf_subsets:
            for split in self.eval_splits:
                queries = self.dataset[subset][split]["queries"]
                combined_text = [
                    f"{title}\n{description}" if description else title
                    for title, description in zip(
                        queries["title"], queries["text"], strict=True
                    )
                ]
                self.dataset[subset][split]["queries"] = queries.remove_columns(
                    ["title", "text"]
                ).add_column("text", combined_text)


class GLAMI1MI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GLAMI1MI2TRetrieval",
        description="Retrieve multilingual fashion product text from product images.",
        reference="https://arxiv.org/abs/2211.14451",
        dataset={
            "path": "artist/glami-1m-t2i-mteb",
            "revision": "20969505f3edaa4f2b239155ec16a4c986b1b195",
        },
        type="Any2AnyMultilingualRetrieval",
        category="i2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=GLAMI1MT2IRetrieval.metadata.eval_langs,
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["E-commerce", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        is_beta=True,
        adapted_from=["GLAMI1MT2IRetrieval"],
        bibtex_citation=GLAMI1MT2IRetrieval.metadata.bibtex_citation,
        prompt={"query": "Find the fashion product matching this image."},
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for subset in self.hf_subsets:
            for split in self.eval_splits:
                data = self.dataset[subset][split]
                relevant_docs = {}
                for text_id, image_scores in data["relevant_docs"].items():
                    for image_id, score in image_scores.items():
                        relevant_docs.setdefault(image_id, {})[text_id] = score
                data["queries"], data["corpus"] = data["corpus"], data["queries"]
                data["relevant_docs"] = relevant_docs
