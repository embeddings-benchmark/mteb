from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class InfoSeekIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InfoSeekIT2TRetrieval",
        description="Retrieve source information to answer questions about images.",
        reference="https://aclanthology.org/2023.emnlp-main.925",
        dataset={
            "path": "mteb/mbeir_infoseek_task6",
            "revision": "4510aa3b456b23f39564694e053e70492cc1de9f",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen2023can,
  author = {Chen, Yang and Hu, Hexiang and Luan, Yi and Sun, Haitian and Changpinyo, Soravit and Ritter, Alan and Chang, Ming-Wei},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {14948--14968},
  title = {Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?},
  year = {2023},
}
""",
        prompt={
            "query": "Find a paragraph from Wikipedia that answers my question about this image."
        },
        superseded_by="InfoSeekIT2TRetrieval.v2",
    )


class InfoSeekIT2TRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InfoSeekIT2TRetrieval.v2",
        description=(
            "Retrieve source information to answer questions about images. "
            "Version 2 sets the canonical metric to hit_rate_at_5, matching the "
            "M-BEIR/UniIR source benchmark's hit-style Recall@5, instead of "
            "ndcg_at_10. Dataset, corpus, and qrels are identical to "
            "InfoSeekIT2TRetrieval. See "
            "[Issue #5214](https://github.com/embeddings-benchmark/mteb/issues/5214)."
        ),
        reference="https://aclanthology.org/2023.emnlp-main.925",
        dataset={
            "path": "mteb/mbeir_infoseek_task6",
            "revision": "4510aa3b456b23f39564694e053e70492cc1de9f",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen2023can,
  author = {Chen, Yang and Hu, Hexiang and Luan, Yi and Sun, Haitian and Changpinyo, Soravit and Ritter, Alan and Chang, Ming-Wei},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages = {14948--14968},
  title = {Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?},
  year = {2023},
}
""",
        prompt={
            "query": "Find a paragraph from Wikipedia that answers my question about this image."
        },
        adapted_from=["InfoSeekIT2TRetrieval"],
    )
