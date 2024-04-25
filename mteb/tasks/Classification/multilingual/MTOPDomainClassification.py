from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "th": ["tha-Thai"],
}


class MTOPDomainClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="MTOPDomainClassification",
        dataset={
            "path": "mteb/mtop_domain",
            "revision": "d80d48c1eb48d3562165c59d59d0034df9fff0bf",
        },
        description="MTOP: Multilingual Task-Oriented Semantic Parsing",
        reference="https://arxiv.org/pdf/2008.09335.pdf",
        category="s2s",
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
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
        n_samples={"validation": 2235, "test": 4386},
        avg_character_length={"validation": 36.5, "test": 36.8},
    )
