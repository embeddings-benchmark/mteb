from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonPolarityVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonPolarityVNClassification",
        description="A collection of translated Amazon customer reviews annotated for polarity classification. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://huggingface.co/datasets/amazon_polarity",
        dataset={
            "path": "GreenNode/amazon-polarity-vn",
            "revision": "4e9a0d6e6bd97ab32f23c50c043d751eed2a5f8a",
        },
        type="Classification",
        category="t2c",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        bibtex_citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
        adapted_from=["AmazonPolarityClassification"],
    )
