from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBWikimQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBWikimQARetrieval",
        dataset={
            "path": "mteb/LEMBWikimQARetrieval",
            "revision": "4e042f63501bf23b3905198bf90d11d3baa12cf0",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("2wikimqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2019-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ho2020constructing,
  author = {Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  pages = {6609--6625},
  title = {Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
  year = {2020},
}
""",
    )
