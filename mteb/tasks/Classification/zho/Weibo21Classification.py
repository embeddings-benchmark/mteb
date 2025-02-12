from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Weibo21Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Weibo21Classification",
        description="Fake news detection in finance domain.",
        reference="https://dl.acm.org/doi/pdf/10.1145/3459637.3482139",
        dataset={
            "path": "FinanceMTEB/MDFEND-Weibo21",
            "revision": "db799d3d74bc752cb30b264a6254ab52471f693d",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=("2014-12-01", "2021-03-31"),
        domains=["Finance"],
        license="acm",
        annotations_creators="expert-annotated",
        bibtex_citation="""@inproceedings{Nan_2021, series={CIKM ’21},
           title={MDFEND: Multi-domain Fake News Detection},
           url={http://dx.doi.org/10.1145/3459637.3482139},
           DOI={10.1145/3459637.3482139},
           booktitle={Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
           publisher={ACM},
           author={Nan, Qiong and Cao, Juan and Zhu, Yongchun and Wang, Yanyan and Li, Jintao},
           year={2021},
           month=oct, pages={3343–3347},
           collection={CIKM ’21} }
""",
        descriptive_stats={
            "num_samples": {"test": 238},
            "average_text_length": {"test": 155.01260504201682},
            "unique_labels": {"test": 2},
            "labels": {"test": {"0": {"count": 168}, "1": {"count": 70}}},
        },
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
