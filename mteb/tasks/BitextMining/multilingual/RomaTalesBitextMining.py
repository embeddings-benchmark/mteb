from __future__ import annotations

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class RomaTalesBitextMining(AbsTaskBitextMining):
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
        eval_langs=["rom-Latn", "hun-Latn"],
        main_score="f1",
        date=None,  # Unknown, these are folk tales
        form=["written"],
        domains=["Fiction"],
        task_subtypes=None,  # Didn't fit any
        license="Not specified",
        socioeconomic_status="low",
        annotations_creators="expert-annotated",
        dialect=["Lovari"],
        text_creation="created",
        bibtex_citation=None,
        n_samples={"test": 215},
        avg_character_length={"test": 316.8046511627907},
    )

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("romani", "sentence1")
        self.dataset = self.dataset.rename_column("hungarian", "sentence2")
