from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.abs_text_sts import AbsTextSTS


class RuSTSBenchmarkSTS(AbsTextSTS):
    metadata = TaskMetadata(
        name="RuSTSBenchmarkSTS",
        dataset={
            "path": "ai-forever/ru-stsbenchmark-sts",
            "revision": "7cf24f325c6da6195df55bef3d86b5e0616f3018",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Russian and verified. "
        + "The dataset was checked with RuCOLA model to ensure that the translation is good and filtered.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2018-01-01"),
        domains=["News", "Social", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""@InProceedings{huggingface:dataset:stsb_multi_mt,
title = {Machine translated multilingual STS benchmark dataset.},
author={Philip May},
year={2021},
url={https://github.com/PhilipMay/stsb-multi-mt}
}""",
    )

    min_score = 0
    max_score = 5
