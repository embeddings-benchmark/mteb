import datasets

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class BornholmBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata()TaskMetadata(
        name="BornholmBitextMining",
        hf_hub_name="strombergnlp/bornholmsk_parallel",
        description="Danish Bornholmsk Parallel Corpus. Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.",
        reference="https://aclanthology.org/W19-6138/",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da", "da-bornholm"],
        main_score="f1",
        revision="3bc5cfb4ec514264fe2db5615fac9016f7251552",
        date=None,
        form=None,
        domains=None,
        license="",
        task_subtypes=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

        @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return

    def load_data(self, **kwargs):
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

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("da", "sentence1")
        self.dataset = self.dataset.rename_column("da_bornholm", "sentence2")
