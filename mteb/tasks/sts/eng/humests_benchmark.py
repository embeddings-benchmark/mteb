from mteb.abstasks import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class HUMESTSBenchmark(AbsTaskSTS):
    metadata = TaskMetadata(
        name="HUMESTSBenchmark",
        dataset={
            "path": "mteb/mteb-human-stsbenchmark-sts",
            "revision": "cb05d5409f802e68d6ed39615ed67f7dc2235ac5",
        },
        description="Human evaluation subset of Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{huggingface:dataset:stsb_multi_mt,
  author = {Philip May},
  title = {Machine translated multilingual STS benchmark dataset.},
  url = {https://github.com/PhilipMay/stsb-multi-mt},
  year = {2021},
}
""",
        adapted_from=["STSBenchmark"],
    )

    min_score = 0
    max_score = 5
