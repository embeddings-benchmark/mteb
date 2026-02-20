from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ArguAna(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAna",
        description="ArguAna: Retrieval of the Best Counterargument without Prior Topic Knowledge",
        reference="http://argumentation.bplaced.net/arguana/data",
        dataset={
            "path": "mteb/arguana",
            "revision": "c22ab2a51041ffd869aaddef7af8d8215647e41a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-07-01"],  # best guess: based on publication date
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wachsmuth2018retrieval,
  author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
  booktitle = {ACL},
  title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
  year = {2018},
}
""",
        prompt={"query": "Given a claim, find documents that refute the claim"},
    )
