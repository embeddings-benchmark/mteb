import datasets

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining


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
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={"rom-hun": ["rom-Latn", "hun-Latn"]},
        main_score="f1",
        date=(
            "1800-01-01",
            "1950-12-31",
        ),  # Broad historical range for the creation of folk tales
        domains=["Fiction", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=["Lovari"],
        sample_creation="created",
        bibtex_citation="",
    )

    def load_data(self) -> None:
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(**self.metadata.dataset)

        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        for lang in self.hf_subsets:
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"romani": "sentence1", "hungarian": "sentence2"}
            )
