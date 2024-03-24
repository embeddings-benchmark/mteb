from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class ATEC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="ATEC",
        hf_hub_name="C-MTEB/ATEC",
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="0f319b1142f2ae3f7dc7be10c3c7f3598ec6c602",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class BQ(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BQ",
        hf_hub_name="C-MTEB/BQ",
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="e3dda5e115e487b39ec7e618c0c6a29137052a55",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class LCQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="LCQMC",
        hf_hub_name="C-MTEB/LCQMC",
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="17f9b096f80380fce5ed12a9be8be7784b337daf",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class PAWSX(AbsTaskSTS):
    metadata = TaskMetadata(
        name="PAWSX",
        hf_hub_name="C-MTEB/PAWSX",
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="9c6a90e430ac22b5779fb019a23e820b11a8b5e1",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class STSB(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSB",
        hf_hub_name="C-MTEB/STSB",
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="0cde68302b3541bb8b3c340dc0644b0b745b3dc0",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict


class AFQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="AFQMC",
        hf_hub_name="C-MTEB/AFQMC",
        description="A Chinese dataset for textual relatedness",
        reference="https://aclanthology.org/2021.emnlp-main.357",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="b44c3b011063adb25877c13823db83bb193913c4",
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
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict


class QBQTC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="QBQTC",
        hf_hub_name="C-MTEB/QBQTC",
        description="",
        reference="https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        revision="790b0510dc52b1553e8c49f3d2afb48c0e5c48b7",
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
