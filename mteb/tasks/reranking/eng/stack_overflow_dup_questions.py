from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class StackOverflowDupQuestions(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StackOverflowDupQuestions",
        description="Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        dataset={
            "path": "mteb/StackOverflowDupQuestions",
            "revision": "5debda000fe8e27ebb5c123d38081f92e1847a59",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2014-01-21", "2018-01-01"),
        domains=["Written", "Blog", "Programming"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve duplicate questions from StackOverflow forum",
        bibtex_citation=r"""
@article{Liu2018LinkSOAD,
  author = {Xueqing Liu and Chi Wang and Yue Leng and ChengXiang Zhai},
  journal = {Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering},
  title = {LinkSO: a dataset for learning to retrieve similar question answer pairs on software development forums},
  url = {https://api.semanticscholar.org/CorpusID:53111679},
  year = {2018},
}
""",
    )
