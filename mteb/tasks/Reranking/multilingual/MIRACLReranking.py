from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskReranking import AbsTaskReranking


class MIRACLReranking(MultilingualTask, AbsTaskReranking):
    metadata = TaskMetadata(
        name="MIRACLReranking",
        description="MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages. This task focuses on the German and Spanish subset.",
        reference="https://project-miracl.github.io/",
        dataset={
            "path": "jinaai/miracl",
            "revision": "d28a029f35c4ff7f616df47b0edf54e6882395e6",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "de": ["deu-Latn"],
            "es": ["spa-Latn"],
        },
        main_score="map",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
