from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class SickFrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICKFr",
        dataset={
            "path": "Lajavaness/SICK-fr",
            "revision": "e077ab4cf4774a1e36d86d593b150422fafd8e8a",
        },
        description="SICK dataset french version",
        reference="https://huggingface.co/datasets/Lajavaness/SICK-fr",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["fra-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=[],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {
                "sentence_A": "sentence1",
                "sentence_B": "sentence2",
                "relatedness_score": "score",
                "Unnamed: 0": "id",
            }
        )
