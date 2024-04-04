from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class EmotionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="EmotionClassification",
        description="Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.",
        reference="https://www.aclweb.org/anthology/D18-1404",
        dataset={
            "path": "mteb/emotion",
            "revision": "4f58c6b202a23cf9a4da393831edf4f9183cad37",
        },
        type="Classification",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["en"],
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
        n_samples={"validation": 2000, "test": 2000},
        avg_character_length={"validation": 95.3, "test": 95.6},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 10
        metadata_dict["samples_per_label"] = 16
        return metadata_dict
