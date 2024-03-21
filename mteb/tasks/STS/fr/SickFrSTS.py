from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class SickFrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICKFr",
        hf_hub_name="Lajavaness/SICK-fr",
        description="SICK dataset french version",
        reference="https://huggingface.co/datasets/Lajavaness/SICK-fr",
        type="STS",
        category="s2s",
        eval_splits=["validation", "test"],
        eval_langs=["fr"],
        main_score="cosine_spearman",
        revision="e077ab4cf4774a1e36d86d593b150422fafd8e8a",
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
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub and rename columns to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision", None),
        )

        self.dataset = self.dataset.rename_columns(
            {
                "sentence_A": "sentence1",
                "sentence_B": "sentence2",
                "relatedness_score": "score",
                "Unnamed: 0": "id",
            }
        )
        self.data_loaded = True
