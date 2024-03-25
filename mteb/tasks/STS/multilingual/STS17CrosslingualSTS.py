from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, CrosslingualTask

_LANGUAGES = [
    "ko-ko",
    "ar-ar",
    "en-ar",
    "en-de",
    "en-en",
    "en-tr",
    "es-en",
    "es-es",
    "fr-en",
    "it-en",
    "nl-en",
]


class STS17Crosslingual(AbsTaskSTS, CrosslingualTask):
    metadata = TaskMetadata(
        name="STS17",
        hf_hub_name="mteb/sts17-crosslingual-sts",
        description="STS 2017 dataset",
        reference="http://alt.qcri.org/semeval2016/task1/",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        revision="af5e6fb845001ecf41f4c1e033ce921939a2a68d",
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
        n_samples={"test": 500},
        avg_character_length={"test": 43.3},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
