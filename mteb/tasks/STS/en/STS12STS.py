from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS12STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS12",
        dataset={
            "path": "mteb/sts12-sts",
            "revision": "a0d554a64d88156834ff5ae9920b964011b16384",
        },
        description="SemEval STS 2012 dataset.",
        reference="https://www.aclweb.org/anthology/S12-1051.pdf",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["en"],
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
