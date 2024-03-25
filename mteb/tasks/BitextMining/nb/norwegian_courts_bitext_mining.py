from typing import Any

import datasets

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class NorwegianCourtsBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="NorwegianCourtsBitextMining",
        hf_hub_name="kaedrodrur/norwegian-courts",
        description="Nynorsk and Bokmål parallel corpus from Norwegian courts. ",
        reference="https://opus.nlpl.eu/ELRC-Courts_Norway-v1.php",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nb", "nn"],
        main_score="accuracy",
        revision="d79af07e969a6678fcbbe819956840425816468f",
        date=("2000-01-01", "2020-12-31"),  # approximate guess
        form=["spoken"],
        domains=["Spoken"],
        task_subtypes=["Political classification"],
        license="openUnder-PSI",
        socioeconomic_status="high",
        annotations_creators="derived",  # best guess
        dialect=[],
        text_creation="found",
        bibtex_citation=None,
        n_samples={"test": 456},
        avg_character_length={"test": 82.11},
    )

    def load_data(self, **kwargs: Any) -> None:  # noqa: ARG002
        """
        Load dataset from HuggingFace hub and convert it to the standard format.
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision", None),
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        # Convert to standard format
        self.dataset = self.dataset.rename_column("nb", "sentence1")
        self.dataset = self.dataset.rename_column("nn", "sentence2")
