from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class GermanSTSBenchmarkSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="GermanSTSBenchmark",
        dataset={
            "path": "jinaai/german-STSbenchmark",
            "revision": "e36907544d44c3a247898ed81540310442329e20",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. "
        "Translations were originally done by T-Systems on site services GmbH.",
        reference="https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["deu-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@InProceedings{huggingface:dataset:stsb_multi_mt,
title = {Machine translated multilingual STS benchmark dataset.},
author={Philip May},
year={2021},
url={https://github.com/PhilipMay/stsb-multi-mt}
}""",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
