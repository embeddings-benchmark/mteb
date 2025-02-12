from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinNSPClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FinNSPClassification",
        description="Financial negative news and its subject determination dataset.",
        reference="https://github.com/alipay/financial_evaluation_dataset/",
        dataset={
            "path": "FinanceMTEB/FinNSP",
            "revision": "1d3ae2b90b692ca702a76f26b94c7cb09b23ca14",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        modalities=["text"],
        date=("2003-02-01", "2023-02-26"),
        domains=["Finance"],
        license="cc-by-4.0",
        annotations_creators="derived",
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
            "average_text_length": {"test": 216.999},
            "unique_labels": {"test": 2},
            "labels": {"test": {"0": {"count": 461}, "1": {"count": 539}}},
        },
    )
