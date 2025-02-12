from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinChinaSentimentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinChinaSentimentClassification",
        description="Polar sentiment dataset of sentences from financial domain, categorized by sentiment into positive, negative, or neutral.",
        reference="https://arxiv.org/abs/2306.14096",
        dataset={
            "path": "FinanceMTEB/FinChinaSentiment",
            "revision": "97eef2264cdadab25f5ba218355e75cb7b4d44ef",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=("2023-06-23", "2023-09-15"),
        domains=["Finance"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        bibtex_citation="""@misc{lu2023bbtfincomprehensiveconstructionchinese,
              title={BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark},
              author={Dakuan Lu and Hengkui Wu and Jiaqing Liang and Yipei Xu and Qianyu He and Yipeng Geng and Mengkun Han and Yingsi Xin and Yanghua Xiao},
              year={2023},
              eprint={2302.09432},
              archivePrefix={arXiv},
              primaryClass={cs.CL},
              url={https://arxiv.org/abs/2302.09432},
        }""",
        descriptive_stats={
            "num_samples": {"test": 1000},
            "average_text_length": {"test": 1202.622},
            "unique_labels": {"test": 4},
            "labels": {
                "test": {
                    "-1": {"count": 762},
                    "-2": {"count": 118},
                    "0": {"count": 102},
                    "-3": {"count": 18},
                }
            },
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
