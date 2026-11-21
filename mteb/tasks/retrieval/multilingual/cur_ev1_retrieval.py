from __future__ import annotations

from enum import StrEnum

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn", "eng-Latn"],
    "es": ["spa-Latn", "eng-Latn"],
    "fr": ["fra-Latn", "eng-Latn"],
}


class CUREv1Splits(StrEnum):
    all = "All"
    dentistry_and_oral_health = "Dentistry and Oral Health"
    dermatology = "Dermatology"
    gastroenterology = "Gastroenterology"
    genetics = "Genetics"
    neuroscience_and_neurology = "Neuroscience and Neurology"
    orthopedic_surgery = "Orthopedic Surgery"
    otorhinolaryngology = "Otorhinolaryngology"
    plastic_surgery = "Plastic Surgery"
    psychiatry_and_psychology = "Psychiatry and Psychology"
    pulmonology = "Pulmonology"

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._member_names_)


class CUREv1Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "mteb/CUREv1",
            "revision": "d974119fc64be8de6337e9a467b900e7e5fa5eac",
        },
        name="CUREv1",
        description="Collection of query-passage pairs curated by medical professionals, across 10 disciplines and 3 cross-lingual settings.",
        type="Retrieval",
        modalities=["text"],
        category="t2t",
        reference="https://huggingface.co/datasets/clinia/CUREv1",
        eval_splits=CUREv1Splits.names(),
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-10-31"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt={
            "query": "Given a question by a medical professional, retrieve relevant passages that best answer the question",
        },
    )
