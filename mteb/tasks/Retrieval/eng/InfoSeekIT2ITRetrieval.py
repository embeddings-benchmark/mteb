from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class InfoSeekIT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InfoSeekIT2ITRetrieval",
        description="Retrieve source text and image information to answer questions about images.",
        reference="https://aclanthology.org/2023.emnlp-main.925",
        dataset={
            "path": "mteb/InfoSeekIT2ITRetrieval",
            "revision": "3baeebfd742e4130d2a75a842740ebb31ea4e6e9",
        },
        type="Any2AnyRetrieval",
        category="it2it",
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
            "query": "Find an image and subject description from Wikipedia that answers my question about this image."
        },
    )
