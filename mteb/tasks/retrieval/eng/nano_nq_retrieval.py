from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tasks.retrieval.eng.nq_retrieval import NQ_CITATION


class NanoNQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNQRetrieval",
        description="NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.",
        reference="https://ai.google.com/research/NaturalQuestions",
        dataset={
            "path": "mteb/NanoNQRetrieval",
            "revision": "e3973b405feb9e94d07fd971d690d24d7f45264a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2019-01-01", "2019-12-31"],
        domains=["Academic", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=NQ_CITATION,
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NQ"],
    )
