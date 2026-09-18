from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoArguAnaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoArguAnaRetrieval",
        description="NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.",
        reference="http://argumentation.bplaced.net/arguana/data",
        dataset={
            "path": "mteb/NanoArguAnaRetrieval",
            "revision": "63ac9f026e8f6b85dc0622f3d630ed7a1758e9ec",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-4.0",
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
        adapted_from=["ArguAna"],
    )
