from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"

_LANGS = {
    # <iso_639_3>-<ISO_15924>
    "arabic": ["ara-Arab"],
    "chinese": ["zho-Hans"],
    "english": ["eng-Latn"],
    "french": ["fra-Latn"],
    "korean": ["kor-Hang"],
    "russian": ["rus-Cyrl"],
    "spanish": ["spa-Latn"],
    "vietnamese": ["vie-Latn"],
}


class PublicHealthQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PublicHealthQA",
        description="A multilingual dataset for public health question answering, based on FAQ sourced from CDC and WHO.",
        dataset={
            "path": "mteb/PublicHealthQA",
            "revision": "60819f7b1cbb998d3b7ccc78fe0aff4198d4b69e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        reference="https://huggingface.co/datasets/xhluca/publichealth-qa",
        date=("2020-01-01", "2020-04-15"),
        domains=["Medical", "Government", "Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{xing_han_lu_2024,
  author = { {Xing Han Lu} },
  doi = { 10.57967/hf/2247 },
  publisher = { Hugging Face },
  title = { publichealth-qa (Revision 3b67b6b) },
  url = { https://huggingface.co/datasets/xhluca/publichealth-qa },
  year = {2024},
}
""",
    )
