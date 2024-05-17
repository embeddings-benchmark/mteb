from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, CrosslingualTask

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "pl": ["pol-Latn"],
    "tr": ["tur-Latn"],
    "ar": ["ara-Arab"],
    "ru": ["rus-Cyrl"],
    "zh": ["cmn-Hans"],
    "fr": ["fra-Latn"],
    "de-en": ["deu-Latn", "eng-Latn"],
    "es-en": ["spa-Latn", "eng-Latn"],
    "it": ["ita-Latn"],
    "pl-en": ["pol-Latn", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
    "es-it": ["spa-Latn", "ita-Latn"],
    "de-fr": ["deu-Latn", "fra-Latn"],
    "de-pl": ["deu-Latn", "pol-Latn"],
    "fr-pl": ["fra-Latn", "pol-Latn"],
}


class STS22CrosslingualSTS(AbsTaskSTS, CrosslingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="STS22",
        dataset={
            "path": "mteb/sts22-crosslingual-sts",
            "revision": "de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3",
        },
        description="SemEval 2022 Task 8: Multilingual News Article Similarity",
        reference="https://competitions.codalab.org/competitions/33835",
        type="STS",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
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
        n_samples={"test": 8060},
        avg_character_length={"train": 1992.8},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 4
        return metadata_dict
