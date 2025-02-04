from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PubChemAISentenceParaphrasePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PubChemAISentenceParaphrasePC",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/PubChemAISentenceParaphrasePC",
            "revision": "f33a205966ce032f957c3a22f4f9e378f89a2c56",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""
        @article{kasmaee2024chemteb,
        title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
        author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
        journal={arXiv preprint arXiv:2412.00532},
        year={2024}
        }
        @article{kim2023pubchem,
        title={PubChem 2023 update},
        author={Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and others},
        journal={Nucleic acids research},
        volume={51},
        number={D1},
        pages={D1373--D1380},
        year={2023},
        publisher={Oxford University Press}
        }
        """,
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            _dataset[split] = [
                {
                    "sentence1": hf_dataset["sent1"],
                    "sentence2": hf_dataset["sent2"],
                    "labels": hf_dataset["labels"],
                }
            ]
        self.dataset = _dataset
