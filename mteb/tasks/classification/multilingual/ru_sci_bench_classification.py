from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class RuSciBenchCoreRiscClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchCoreRiscClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="This binary classification task aims to determine whether a scientific paper (based on its title and abstract) belongs to the Core of the Russian Science Citation Index (RISC). The RISC includes a wide range of publications, but the Core RISC comprises the most cited and prestigious journals, dissertations, theses, monographs, and studies. The task is provided for both Russian and English versions of the paper's title and abstract.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="t2c",
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
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Classify whether a scientific article is part of the core RISC or not based on the title and abstract",
    )


class RuSciBenchPubTypeClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchPubTypeClassification",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="This task involves classifying scientific papers (based on their title and abstract) into different publication types. The dataset identifies the following types: 'Article', 'Conference proceedings', 'Survey', 'Miscellanea', 'Short message', 'Review', and 'Personalia'. This task is available for both Russian and English versions of the paper's title and abstract.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="t2c",
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
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Classify the type of scientific paper based on the title and abstract",
    )


class RuSciBenchGRNTIClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchGRNTIClassification.v2",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="Classification of scientific papers based on the GRNTI (State Rubricator of Scientific and Technical Information) rubricator. GRNTI is a universal hierarchical classification of knowledge domains adopted in Russia and CIS countries to systematize the entire flow of scientific and technical information. This task uses the first level of the GRNTI hierarchy and top 28 classes by frequency. In this version, English language support has been added and data partitioning has been slightly modified.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "grnti_ru": ["rus-Cyrl"],
            "grnti_en": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Classify the category of scientific papers based on the titles and abstracts",
    )


class RuSciBenchOECDClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuSciBenchOECDClassification.v2",
        dataset={
            "path": "mlsa-iai-msu-lab/ru_sci_bench_mteb",
            "revision": "fbc0599a0b5f00b3c7d87ab4d13490f04fb77f8e",
        },
        description="Classification of scientific papers based on the OECD (Organization for Economic Co-operation and Development) rubricator. OECD provides a hierarchical 3-level system of classes for labeling scientific articles. This task uses the first two levels of the OECD hierarchy, top 29 classes. In this version, English language support has been added and data partitioning has been slightly modified.",
        reference="https://github.com/mlsa-iai-msu-lab/ru_sci_bench_mteb",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "oecd_ru": ["rus-Cyrl"],
            "oecd_en": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2007-01-01", "2023-01-01"),
        domains=["Academic", "Non-fiction", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{vatolin2024ruscibench,
  author = {Vatolin, A. and Gerasimenko, N. and Ianina, A. and Vorontsov, K.},
  doi = {10.1134/S1064562424602191},
  issn = {1531-8362},
  journal = {Doklady Mathematics},
  month = {12},
  number = {1},
  pages = {S251--S260},
  title = {RuSciBench: Open Benchmark for Russian and English Scientific Document Representations},
  url = {https://doi.org/10.1134/S1064562424602191},
  volume = {110},
  year = {2024},
}
""",
        prompt="Classify the category of scientific papers based on the titles and abstracts",
    )
