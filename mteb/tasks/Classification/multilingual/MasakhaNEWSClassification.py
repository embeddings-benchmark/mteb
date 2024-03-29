from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = [
    "amh",
    "eng",
    "fra",
    "hau",
    "ibo",
    "lin",
    "lug",
    "orm",
    "pcm",
    "run",
    "sna",
    "som",
    "swa",
    "tir",
    "xho",
    "yor",
]


class MasakhaNEWSClassification(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="MasakhaNEWSClassification",
        dataset={
            "path": "masakhane/masakhanews",
            "revision": "8ccc72e69e65f40c70e117d8b3c08306bb788b60",
        }
        description="MasakhaNEWS is the largest publicly available dataset for news topic classification in 16 languages widely spoken in Africa. The train/validation/test sets are available for all the 16 languages.",
        reference="https://arxiv.org/abs/2304.09972",
        category="s2s",
        type="Classification",
        eval_splits=["test"],
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
        n_samples={"test": 422},
        avg_character_length={"test": 5116.6},
    )
