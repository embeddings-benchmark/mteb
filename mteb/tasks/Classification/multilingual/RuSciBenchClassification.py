from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class RuSciBenchCoreRiscClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchCoreRiscClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="Classification of scientific papers (title+abstract) by publication Core RISC status",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "corerisc_ru": ["rus-Cyrl"],
            "corerisc_en": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
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
        prompt="Classify whether a scientific article is part of the core RISC or not based on the title and abstract",
    )


class RuSciBenchPubTypeClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchPubTypeClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="Classification of scientific papers (title+abstract) by publication type",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "pub_type_ru": ["rus-Cyrl"],
            "pub_type_en": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
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
        prompt="Classify the type of scientific paper based on the title and abstract",
    )
