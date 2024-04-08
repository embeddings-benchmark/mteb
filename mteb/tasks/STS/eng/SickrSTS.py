from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class SickrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICK-R",
        dataset={
            "path": "mteb/sickr-sts",
            "revision": "20a6d6f312dd54037fe07a32d58e5e168867909d",
        },
        description="Semantic Textual Similarity SICK-R dataset as described here:",
        reference="https://aclanthology.org/2020.lrec-1.207",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
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
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
