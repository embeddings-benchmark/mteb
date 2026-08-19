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
            "Version 2 restores the combined InfoSeek + OVEN candidate corpus "
            "used by the original M-BEIR benchmark. Queries and qrels are "
            "unchanged. For more information see "
            "[Issue #5021](https://github.com/embeddings-benchmark/mteb/issues/5021)."
        ),
        reference="https://aclanthology.org/2023.emnlp-main.925",
        dataset={
            "path": "lxercode/mbeir_infoseek_task6_v2",
            "revision": "2d68d6828333bedf9749af0c20269a9c15e64997",
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
        adapted_from=["InfoSeekIT2TRetrieval"],
    )
