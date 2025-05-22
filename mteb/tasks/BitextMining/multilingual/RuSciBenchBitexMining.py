from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "en-ru": ["rus-Cyrl", "rus-Cyrl"],
}


_SPLITS = ["test"]


class RuSciBenchBitexMining(AbsTaskBitextMining, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="RuSciBenchBitexMining",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_bitext_mining",
            "revision": "927d95897a168b79568e96591276b995ed1c4da8",
        },
        description="Find translation of a scientific article",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="BitextMining",
        category="p2p",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="not specified",
        dialect=[],
        sample_creation="found",
        annotations_creators="derived",
        bibtex_citation="""
@article{vatolin2024ruscibench,
  author  = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  title   = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  journal = {Doklady Mathematics},
  year    = {2024},
  volume  = {110},
  number  = {1},
  pages   = {S251--S260},
  month   = {12},
  doi     = {10.1134/S1064562424602191},
  url     = {https://doi.org/10.1134/S1064562424602191},
  issn    = {1531-8362}
}""",
        prompt="Given the following title and abstract of the scientific article, find its translation",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset.setdefault(lang, {})[_SPLITS[0]] = datasets.load_dataset(
                split=_SPLITS[0],
                name=lang,
                **self.metadata_dict["dataset"],
            )

        self.dataset_transform()
        self.data_loaded = True
