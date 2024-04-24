from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class ATEC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="ATEC",
        dataset={
            "path": "C-MTEB/ATEC",
            "revision": "0f319b1142f28d00e055a6770f3f726ae9b7d865",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
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
        metadata_dict["max_score"] = 1
        return metadata_dict


class BQ(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BQ",
        dataset={
            "path": "C-MTEB/BQ",
            "revision": "e3dda5e115e487b39ec7e618c0c6a29137052a55",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
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
        metadata_dict["max_score"] = 1
        return metadata_dict


class LCQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="LCQMC",
        dataset={
            "path": "C-MTEB/LCQMC",
            "revision": "17f9b096f80380fce5ed12a9be8be7784b337daf",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
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
        metadata_dict["max_score"] = 1
        return metadata_dict


class PAWSX(AbsTaskSTS):
    metadata = TaskMetadata(
        name="PAWSX",
        dataset={
            "path": "C-MTEB/PAWSX",
            "revision": "9c6a90e430ac22b5779fb019a23e820b11a8b5e1",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
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
        metadata_dict["max_score"] = 1
        return metadata_dict


class STSB(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSB",
        dataset={
            "path": "C-MTEB/STSB",
            "revision": "0cde68302b3541bb8b3c340dc0644b0b745b3dc0",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
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


class AFQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="AFQMC",
        dataset={
            "path": "C-MTEB/AFQMC",
            "revision": "b44c3b011063adb25877c13823db83bb193913c4",
        },
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
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
        metadata_dict["max_score"] = 1
        return metadata_dict


class QBQTC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="QBQTC",
        dataset={
            "path": "C-MTEB/QBQTC",
            "revision": "790b0510dc52b1553e8c49f3d2afb48c0e5c48b7",
        },
        description="",
        reference="https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
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
