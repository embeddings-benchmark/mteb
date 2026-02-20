from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RuBQReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RuBQReranking",
        dataset={
            "path": "mteb/RuBQReranking",
            "revision": "e8233e2234f8b24ab47f203b69d1161c3c0bc5a1",
        },
        description="Paragraph reranking based on RuBQ 2.0. Give paragraphs that answer the question higher scores.",
        reference="https://openreview.net/pdf?id=P5UQFFoQ4PJ",
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="map_at_1000",
        date=("2001-01-01", "2021-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{RuBQ2021,
  author = {Ivan Rybin and Vladislav Korablinov and Pavel Efimov and Pavel Braslavski},
  booktitle = {ESWC},
  pages = {532--547},
  title = {RuBQ 2.0: An Innovated Russian Question Answering Dataset},
  year = {2021},
}
""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question.",
        },
    )
