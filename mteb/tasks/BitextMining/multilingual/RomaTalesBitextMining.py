from __future__ import annotations

import datasets

from mteb.abstasks import AbsTaskBitextMining, CrosslingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class RomaTalesBitextMining(AbsTaskBitextMining, CrosslingualTask):
    metadata = TaskMetadata(
        name="RomaTalesBitextMining",
        dataset={
            "path": "kardosdrur/roma-tales",
            "revision": "f4394dbca6845743cd33eba77431767b232ef489",
        },
        description="Parallel corpus of Roma Tales in Lovari with Hungarian translations.",
        reference="https://idoc.pub/documents/idocpub-zpnxm9g35ylv",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs={"rom-hun": ["rom-Latn", "hun-Latn"]},
        main_score="f1",
        date=None,  # Unknown, these are folk tales
        form=["written"],
        domains=["Fiction"],
        task_subtypes=[],  # Didn't fit any
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="expert-annotated",
        dialect=["Lovari"],
        text_creation="created",
        bibtex_citation=None,
        n_samples={"test": 215},
        avg_character_length={"test": 316.8046511627907},
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.langs:
            self.dataset[lang] = datasets.load_dataset(**self.metadata_dict["dataset"])

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        for lang in self.langs:
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"romani": "sentence1", "hungarian": "sentence2"}
            )
