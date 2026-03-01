from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class GermanSTSBenchmarkSTS(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="GermanSTSBenchmark",
        dataset={
            "path": "mteb/GermanSTSBenchmark",
            "revision": "75829b73cccb64bf86a9f098cbc780d37b786091",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. "
        + "Translations were originally done by T-Systems on site services GmbH.",
        reference="https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["deu-Latn"],
        main_score="cosine_spearman",
        date=("2023-11-09", "2024-01-24"),
        domains=["News", "Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{huggingface:dataset:stsb_multi_mt,
  author = {Philip May},
  title = {Machine translated multilingual STS benchmark dataset.},
  url = {https://github.com/PhilipMay/stsb-multi-mt},
  year = {2021},
}
""",
    )
