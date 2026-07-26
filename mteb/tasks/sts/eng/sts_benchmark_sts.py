from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STSBenchmarkSTS(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STSBenchmark",
        dataset={
            "path": "mteb/stsbenchmark-sts",
            "revision": "b0fddb56ed78048fa8b90373c8a3cfc37b684831",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2021-01-01", "2021-12-31"),  # publication year
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
        superseded_by="STSBenchmark.v2",
    )

    min_score = 0
    max_score = 5


class STSBenchmarkSTSV2(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="STSBenchmark.v2",
        dataset={
            "path": "mteb/STSBenchmarkv2",
            "revision": "93b628c3969a75e76727db2b7ee252e53e96268d",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset. This version removes duplicate sentence pairs from the validation and test splits when the same order-insensitive pair appears more than once in an evaluation split or appears in another split.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2021-01-01", "2021-12-31"),  # publication year
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
