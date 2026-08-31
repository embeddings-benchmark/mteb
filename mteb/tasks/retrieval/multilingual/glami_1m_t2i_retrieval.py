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
            "revision": "072397185348006567b5be8e4071b6b196c1be53",
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
