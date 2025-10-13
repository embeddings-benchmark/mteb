from __future__ import annotations

from mteb.abstasks.AbsTaskMultilabelClassification import (
    AbsTaskMultilabelClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class MultiEURLEXSlovakClassification(AbsTaskMultilabelClassification):
    metadata = TaskMetadata(
        name="MultiEURLEXSlovakClassification",
        description="""Multi-label legal document classification dataset from EU law in Slovak.
        Documents are labeled with multiple EUROVOC concepts reflecting their legal topics.
        This dataset enables evaluation of embedding models on Slovak legal text understanding
        and multi-label classification in the legal domain.""",
        reference="https://huggingface.co/datasets/nlpaueb/multi_eurlex",
        dataset={
            "path": "mteb/eurlex-multilingual",
            "name": "sk",
            "revision": "2aea5a6dc8fdcfeca41d0fb963c0a338930bde5c",
        },
        type="MultilabelClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["slk-Latn"],
        main_score="f1",
        date=("2000-01-01", "2019-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@inproceedings{chalkidis-etal-2021-multieurlex,
    title = "{M}ulti{EURLEX} - A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer",
    author = "Chalkidis, Ilias  and
      Fergadiotis, Manos  and
      Androutsopoulos, Ion",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.559",
    doi = "10.18653/v1/2021.emnlp-main.559",
    pages = "6974--6996",
}""",
    )
