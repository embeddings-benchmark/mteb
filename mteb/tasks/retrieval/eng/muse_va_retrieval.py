from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_REFERENCE = "https://huggingface.co/datasets/jiahaomei/MUSE-VA"
_BIBTEX = r"""
@misc{museva2026,
  author = {Mei, Jiahao and others},
  title = {MUSE-VA: Multimodal MUSic Emotion Dataset with Balanced Valence-Arousal},
  howpublished = {Hugging Face dataset jiahaomei/MUSE-VA},
  year = {2026},
}
"""
_DESCRIPTION = (
    "MUSE-VA (Multimodal MUSic Emotion Dataset with Balanced VA) pairs music clips "
    "with emotion-aligned generated images. The test split contains 625 one-to-one "
    "audio–image pairs spanning diverse genres and VA coordinates."
)


class MUSEVAA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MUSEVAA2IRetrieval",
        description=_DESCRIPTION
        + " Queries are music clips and the corpus contains paired images; the goal "
        "is to retrieve the image that matches the music clip.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/MUSE-VA-A2I",
            "revision": "761cd11f21c1dad47a3fa82160ff5a77ee8cb705",
        },
        type="Any2AnyRetrieval",
        category="a2i",
        modalities=["audio", "image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-07-01"),
        domains=["Music", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the image that best matches the emotion and mood of this music clip."
        },
        is_beta=True,
    )


class MUSEVAI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MUSEVAI2ARetrieval",
        description=_DESCRIPTION
        + " Queries are images and the corpus contains music clips; the goal is to "
        "retrieve the music clip that matches the image.",
        reference=_REFERENCE,
        dataset={
            "path": "Wissam42/MUSE-VA-I2A",
            "revision": "84ec3222e6fd1456db6599fcaf335b3f8c80f9d5",
        },
        type="Any2AnyRetrieval",
        category="i2a",
        modalities=["image", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2026-07-01"),
        domains=["Music", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="created",
        bibtex_citation=_BIBTEX,
        prompt={
            "query": "Retrieve the music clip that best matches the emotion and mood of this image."
        },
        is_beta=True,
    )
