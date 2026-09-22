from typing import Any

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

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


class STSBenchmarkMultilingualSTS(AbsTaskSTS):
    fast_loading = True
    metadata = TaskMetadata(
        name="STSBenchmarkMultilingualSTS",
        dataset={
            "path": "mteb/stsb_multi_mt",
            "revision": "29afa2569dcedaaa2fe6a3dcfebab33d28b82e8c",
        },
        description=(
            "Semantic Textual Similarity Benchmark (STSbenchmark) dataset, "
            "but translated using DeepL API."
        ),
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="t2t",
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
        bibtex_citation=r"""
@inproceedings{cer-etal-2017-semeval,
  address = {Vancouver, Canada},
  author = {Cer, Daniel  and
Diab, Mona  and
Agirre, Eneko  and
Lopez-Gazpio, I{\~n}igo  and
Specia, Lucia},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation ({S}em{E}val-2017)},
  doi = {10.18653/v1/S17-2001},
  editor = {Bethard, Steven  and
Carpuat, Marine  and
Apidianaki, Marianna  and
Mohammad, Saif M.  and
Cer, Daniel  and
Jurgens, David},
  month = aug,
  pages = {1--14},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation},
  url = {https://aclanthology.org/S17-2001},
  year = {2017},
}
""",
    )

    min_score = 0
    max_score = 5

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        for lang, subset in self.dataset.items():
            self.dataset[lang] = subset.rename_column("similarity_score", "score")
