from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EDIST2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EDIST2ITRetrieval",
        description="Retrieve news images and titles based on news content.",
        reference="https://aclanthology.org/2023.emnlp-main.297/",
        dataset={
            "path": "mteb/mbeir_edis_task2",
            "revision": "afc3409142cdf1adab05336dd1bc5fd3c0a659dc",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["News"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{liu2023edis,
  author = {Liu, Siqi and Feng, Weixi and Fu, Tsu-Jui and Chen, Wenhu and Wang, William},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {4877--4894},
  title = {EDIS: Entity-Driven Image Search over Multimodal Web Content},
  year = {2023},
}
""",
        prompt={"query": "Identify the news photo for the given caption."},
        superseded_by="EDIST2ITRetrieval.v2",
    )


class EDIST2ITRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EDIST2ITRetrieval.v2",
        description=(
            "Retrieve news images and titles based on news content. "
            "Version 2 sets the canonical metric to hit_rate_at_5, matching the "
            "M-BEIR/UniIR source metric (hit-style Recall@5) instead of "
            "ndcg_at_10. Dataset, corpus, and qrels are identical to EDIST2ITRetrieval. See "
            "[Issue #5214](https://github.com/embeddings-benchmark/mteb/issues/5214)."
        ),
        reference="https://aclanthology.org/2023.emnlp-main.297/",
        dataset={
            "path": "mteb/mbeir_edis_task2",
            "revision": "afc3409142cdf1adab05336dd1bc5fd3c0a659dc",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["News"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{liu2023edis,
  author = {Liu, Siqi and Feng, Weixi and Fu, Tsu-Jui and Chen, Wenhu and Wang, William},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {4877--4894},
  title = {EDIS: Entity-Driven Image Search over Multimodal Web Content},
  year = {2023},
}
""",
        prompt={"query": "Identify the news photo for the given caption."},
        adapted_from=["EDIST2ITRetrieval"],
    )
