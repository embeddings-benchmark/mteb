from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, CrosslingualTask

_LANGUAGES = [
    "en",
    "de",
    "es",
    "pl",
    "tr",
    "ar",
    "ru",
    "zh",
    "fr",
    "de-en",
    "es-en",
    "it",
    "pl-en",
    "zh-en",
    "es-it",
    "de-fr",
    "de-pl",
    "fr-pl",
]


class STS22CrosslingualSTS(AbsTaskSTS, CrosslingualTask):
    metadata = TaskMetadata(
        name="STS22",
        hf_hub_name="mteb/sts22-crosslingual-sts",
        description="SemEval 2022 Task 8: Multilingual News Article Similarity",
        reference="https://competitions.codalab.org/competitions/33835",
        type="STS",
        category="p2p",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        revision="eea2b4fe26a775864c896887d910b76a8098ad3f",
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
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 4
        return metadata_dict
