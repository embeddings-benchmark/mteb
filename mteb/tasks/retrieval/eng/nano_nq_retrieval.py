from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        bibtex_citation=r"""
@article{47761,
  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
and Slav Petrov},
  journal = {Transactions of the Association of Computational
Linguistics},
  title = {Natural Questions: a Benchmark for Question Answering Research},
  year = {2019},
}
""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NQ"],
    )
