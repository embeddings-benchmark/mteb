from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoQuoraRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoQuoraRetrieval",
        description="NanoQuoraRetrieval is a smaller subset of the "
        "QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a"
        " question, find other (duplicate) questions.",
        reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
        dataset={
            "path": "mteb/NanoQuoraRetrieval",
            "revision": "3ee153e5710da83a7f35df0cb1db4a0ad036072a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2017-01-01", "2017-12-31"],
        domains=["Social"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{quora-question-pairs,
  author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
  publisher = {Kaggle},
  title = {Quora Question Pairs},
  url = {https://kaggle.com/competitions/quora-question-pairs},
  year = {2017},
}
""",
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        adapted_from=["QuoraRetrieval"],
    )
