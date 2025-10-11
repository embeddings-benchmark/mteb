from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"

_EVAL_LANGS = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
}


class XMarket(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XMarket",
        description="XMarket",
        reference=None,
        dataset={
            "path": "mteb/XMarket",
            "revision": "07b0172d008ab25b0a9702119f521bf477267090",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{Bonab_2021,
  author = {Bonab, Hamed and Aliannejadi, Mohammad and Vardasbi, Ali and Kanoulas, Evangelos and Allan, James},
  booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
  collection = {CIKM ’21},
  doi = {10.1145/3459637.3482493},
  month = oct,
  publisher = {ACM},
  series = {CIKM ’21},
  title = {Cross-Market Product Recommendation},
  url = {http://dx.doi.org/10.1145/3459637.3482493},
  year = {2021},
}
""",
    )
