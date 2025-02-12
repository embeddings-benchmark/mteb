from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinFEClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinFEClassification",
        description="Financial social media text sentiment categorization dataset.",
        reference="https://arxiv.org/abs/2302.09432",
        dataset={
            "path": "FinanceMTEB/FinFE",
            "revision": "01034e2afdce0f7fa9a51a03aa0fdc1e3d576b05",
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
            "average_text_length": {"test": 20.767},
            "unique_labels": {"test": 3},
            "labels": {
                "test": {
                    "0": {"count": 287},
                    "2": {"count": 462},
                    "1": {"count": 251},
                }
            },
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
