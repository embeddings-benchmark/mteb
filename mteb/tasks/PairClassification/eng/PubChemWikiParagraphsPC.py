from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PubChemWikiParagraphsPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PubChemWikiParagraphsPC",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/PubChemWikiParagraphsPC",
            "revision": "7fb14716e4106b72f51a16e682e5cd2d67e9bd70"
        },
        type="PairClassification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Chemistry"],
        task_subtypes=None,
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
        @article{kasmaee2024chemteb,
        title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
        author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
        journal={arXiv preprint arXiv:2412.00532},
        year={2024}
        }
        """,
    )

    def load_data(self):
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
