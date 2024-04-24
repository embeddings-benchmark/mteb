from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class STS14STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS14",
        dataset={
            "path": "mteb/sts14-sts",
            "revision": "6031580fec1f6af667f0bd2da0a551cf4f0b2375",
        },
        description="SemEval STS 2014 dataset. Currently only the English dataset",
        reference="https://www.aclweb.org/anthology/S14-1002",
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
        bibtex_citation="",
        n_samples=None,
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
