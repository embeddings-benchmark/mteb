from __future__ import annotations

from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
    "nl": ["nld-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "zh": ["cmn-Hans"],
}

_SPLITS = ["dev", "test"]


class STSBenchmarkMultilingualSTS(AbsTaskSTS, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="STSBenchmarkMultilingualSTS",
        dataset={
            "path": "mteb/stsb_multi_mt",
            "revision": "29afa2569dcedaaa2fe6a3dcfebab33d28b82e8c",
        },
        description=(
            "Semantic Textual Similarity Benchmark (STSbenchmark) dataset, "
            + "but translated using DeepL API."
        ),
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2012-01-01", "2017-12-31"),
        domains=["News", "Social", "Web", "Spoken", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation="""@InProceedings{huggingface:dataset:stsb_multi_mt,
        title = {Machine translated multilingual STS benchmark dataset.},
        author={Philip May},
        year={2021},
        url={https://github.com/PhilipMay/stsb-multi-mt}
        }""",
        descriptive_stats={
            "n_samples": {"dev": 30000, "test": 27580},
            "avg_character_length": {"dev": 66.5, "test": 56.1},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self) -> None:
        for lang, subset in self.dataset.items():
            self.dataset[lang] = subset.rename_column("similarity_score", "score")
