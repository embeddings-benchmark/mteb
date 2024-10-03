from __future__ import annotations


from typing import Any

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaAIParagraphsParaphrasePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="WikipediaAIParagraphsParaphrasePC",
        description="""TBW""",
        reference="https://pubchem.ncbi.nlm.nih.gov/",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaAIParagraphsParaphrasePC",
            "revision": "a430437ea6c6fe0e6461e6d6659f647d0bf62496"
        },
        type="PairClassification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_f1",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators="derived",
        dialect=None,
        sample_creation="created",
        bibtex_citation=None,
        descriptive_stats={}
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.metadata_dict["dataset"]["path"],
            revision=self.metadata_dict["dataset"]["revision"],
            trust_remote_code=True,
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sentence1": hf_dataset["sent1"],
                    "sentence2": hf_dataset["sent2"],
                    "labels": hf_dataset["labels"]
                }
            ]
        self.dataset = _dataset
