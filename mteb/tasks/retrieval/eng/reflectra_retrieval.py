from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ReflectraI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ReflectraI2ARetrieval",
        description=(
            "Reflectra evaluates image-to-audio emotion matching. "
            "One thousand images are each rated against six candidate music clips on a 0–10 scale; "
            "scores >= 7 are treated as relevant. "
            "Queries are images and the corpus contains music clips; "
            "the goal is to retrieve music that emotionally matches the image."
        ),
        reference="https://huggingface.co/datasets/AraNge/reflectra-benchmark",
        dataset={
            "path": "Wissam42/Reflectra-I2A",
            "revision": "26d44f3452f3d1be242fa9e6c4d9f8ebfa840af1",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-01"),
        domains=["Music", "Web"],
        task_subtypes=["Cross-Modal Retrieval", "Emotion classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{reflectra2025,
  author = {AraNge},
  title = {Reflectra Benchmark: Image-to-Audio Emotion Matching},
  howpublished = {Hugging Face dataset AraNge/reflectra-benchmark},
  year = {2025},
}
""",
        prompt={"query": "Retrieve music clips that emotionally match this image."},
        is_beta=True,
    )
