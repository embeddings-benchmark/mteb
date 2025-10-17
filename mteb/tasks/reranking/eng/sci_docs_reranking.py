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
@inproceedings{cohan-etal-2020-specter,
  address = {Online},
  author = {Cohan, Arman  and
Feldman, Sergey  and
Beltagy, Iz  and
Downey, Doug  and
Weld, Daniel},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2020.acl-main.207},
  editor = {Jurafsky, Dan  and
Chai, Joyce  and
Schluter, Natalie  and
Tetreault, Joel},
  month = jul,
  pages = {2270--2282},
  publisher = {Association for Computational Linguistics},
  title = {{SPECTER}: Document-level Representation Learning using Citation-informed Transformers},
  url = {https://aclanthology.org/2020.acl-main.207},
  year = {2020},
}
""",
        adapted_from=["SCIDOCS"],
    )
