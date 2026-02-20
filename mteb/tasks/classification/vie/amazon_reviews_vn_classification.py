from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AmazonReviewsVNClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AmazonReviewsVNClassification",
        dataset={
            "path": "GreenNode/amazon-reviews-multi-vn",
            "revision": "27da94deb6d4f44af789a3d70750fa506b79f189",
        },
        description="A collection of translated Amazon reviews specifically designed to aid research in multilingual text classification. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://arxiv.org/abs/2010.02573",
        category="t2c",
        type="Classification",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Reviews", "Written"],
        task_subtypes=["Emotion classification"],
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
        adapted_from=["AmazonReviewsClassification"],
    )
