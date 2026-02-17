from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SciDocsReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SciDocsRR",
        description="Ranking of related scientific papers based on their title.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "mteb/SciDocsRR",
            "revision": "39b8377811871075eed9de3b8a7e21aaa6acb3d8",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=("2000-01-01", "2020-12-31"),  # best guess
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=["Scientific Reranking"],
        license="cc-by-4.0",
        annotations_creators=None,
        dialect=None,
        sample_creation="found",
        prompt="Given a title of a scientific paper, retrieve the titles of other relevant papers",
        bibtex_citation=r"""
@inproceedings{specter2020cohan,
  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle = {ACL},
  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  year = {2020},
}
""",
        adapted_from=["SCIDOCS"],
    )
