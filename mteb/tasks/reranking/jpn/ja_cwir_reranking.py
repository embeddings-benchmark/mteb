from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class JaCWIRReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaCWIRReranking",
        description=(
            "JaCWIR is a small-scale Japanese information retrieval evaluation dataset consisting of "
            "5000 question texts and approximately 500k web page titles and web page introductions or summaries "
            "(meta descriptions, etc.). The question texts are created based on one of the 500k web pages, "
            "and that data is used as a positive example for the question text."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        dataset={
            "path": "mteb/JaCWIRReranking",
            "revision": "48d6b0851fb5ce83b648eb9d3689cf56a2e6d5b1",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="map_at_1000",
        date=("2020-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{yuichi-tateno-2024-jacwir,
  author = {Yuichi Tateno},
  title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
}
""",
    )
