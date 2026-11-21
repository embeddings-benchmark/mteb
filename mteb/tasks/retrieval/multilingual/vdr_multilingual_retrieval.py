from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = {
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "de": ["deu-Latn"],
    "it": ["ita-Latn"],
}
_EVAL_SPLIT = "train"  # It is test split only, but given name as train on HF


class VDRMultilingualRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VDRMultilingualRetrieval",
        description="Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German",
        reference="https://huggingface.co/datasets/llamaindex/vdr-multilingual-test",
        dataset={
            "path": "mteb/VDRMultilingualRetrieval",
            "revision": "37d80c61d6187902b4d823a2f2fb0f18de6b2492",
        },
        type="Retrieval",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=(
            "2025-01-01",
            "2025-01-10",
        ),  # Not Specified exactly in the dataset and blog
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
