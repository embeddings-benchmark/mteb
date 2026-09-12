from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoSCIDOCSRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoSCIDOCSRetrieval",
        description="NanoFiQA2018 is a smaller subset of "
        "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
        " prediction, to document classification and recommendation.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "mteb/NanoSCIDOCSRetrieval",
            "revision": "cf39b8741a6f2e984d0b8f4796b2dd5bb65390f9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2020-01-01", "2020-12-31"],
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{specter2020cohan,
  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle = {ACL},
  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  year = {2020},
}
""",
        prompt={
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
        adapted_from=["SCIDOCS"],
    )
