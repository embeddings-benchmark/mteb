from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class EmotionVNClassification(AbsTaskClassification):
    num_samples = 16

    metadata = TaskMetadata(
        name="EmotionVNClassification",
        description="Emotion is a translated dataset of Vietnamese from English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise. The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://www.aclweb.org/anthology/D18-1404",
        dataset={
            "path": "GreenNode/emotion-vn",
            "revision": "797a93c0dd755ebf5818fbf54d0e0a024df9216d",
        },
        type="Classification",
        category="t2c",
        eval_splits=["validation", "test"],
        eval_langs=["vie-Latn"],
        main_score="accuracy",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Social", "Written"],
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
        adapted_from=["EmotionClassification"],
    )
