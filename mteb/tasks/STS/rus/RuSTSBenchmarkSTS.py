from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class RuSTSBenchmarkSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="RuSTSBenchmarkSTS",
        dataset={
            "path": "ai-forever/ru-stsbenchmark-sts",
            "revision": "7cf24f325c6da6195df55bef3d86b5e0616f3018",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into Russian and verified. "
        "The dataset was checked with RuCOLA model to ensure that the translation is good and filtered.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2018-01-01"),
        form=["written"],
        domains=["News", "Social", "Web"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="machine-translated and verified",
        bibtex_citation="",
        n_samples={"test": 1264},
        avg_character_length={"test": 54.2},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
