from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class JavaneseIMDBClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JavaneseIMDBClassification",
        description="Large Movie Review Dataset translated to Javanese. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets.",
        reference="https://github.com/w11wo/nlp-datasets#javanese-imdb",
        dataset={
            "path": "w11wo/imdb-javanese",
            "revision": "11bef3dfce0ce107eb5e276373dcd28759ce85ee",
            "trust_remote_code": True,
        },
        type="Classification",
        category="s2s",
        date=("2021-06-24", "2021-06-24"),
        eval_splits=["test"],
        eval_langs=["jav-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="MIT",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{wongso2021causal,
            title={Causal and Masked Language Modeling of Javanese Language using Transformer-based Architectures},
            author={Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin},
            booktitle={2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)},
            pages={1--7},
            year={2021},
            organization={IEEE}
        }
        """,
        n_samples={"test": 25_000},
        avg_character_length={"test": 481.83},
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
