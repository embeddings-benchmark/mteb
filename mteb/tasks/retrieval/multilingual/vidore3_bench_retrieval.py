from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGS = {
    "french": ["fra-Latn"],
    "spanish": ["spa-Latn"],
    "english": ["eng-Latn"],
    "german": ["deu-Latn"],
    "italian": ["ita-Latn"],
    "portuguese": ["por-Latn"],
}


class Vidore3FinanceEnRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_finance_en_mteb_format",
            "revision": "fa78cb14152b3dde8c5defdc4e3ddf50de69dfeb",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Financial"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3FinanceEnRetrieval.v2",
    )


class Vidore3FinanceEnRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnRetrieval.v2",
        description="Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceEnOCRRetrieval",
            "revision": "2a88e3e92aea41c8ab9bddaac7983f3b490dcacc",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Financial"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        adapted_from=["Vidore3FinanceEnRetrieval"],
        contributed_by="Illuin Technology",
    )


class Vidore3FinanceFrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_finance_fr_mteb_format",
            "revision": "8a2adfda85a7967c7252129703d9b3c7c9f038a9",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Financial"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3FinanceFrRetrieval.v2",
    )


class Vidore3FinanceFrRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrRetrieval.v2",
        description="Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceFrOCRRetrieval",
            "revision": "f39c5f63623a30d9ab946180e8d698126887e7c4",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Financial"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3FinanceFrRetrieval"],
    )


class Vidore3IndustrialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_industrial_mteb_format",
            "revision": "f732b725cf4a70803210edfe265a04f8bd5328f6",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3IndustrialRetrieval.v2",
    )


class Vidore3IndustrialRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialRetrieval.v2",
        description="Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3IndustrialOCRRetrieval",
            "revision": "ff40e351f82d26dc8b406edf13b6471f00e378d0",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3IndustrialRetrieval"],
    )


class Vidore3PharmaceuticalsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_pharmaceuticals_mteb_format",
            "revision": "237ed4f43c7fb3c4df07ec4e9dd0a4366be555b0",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Medical"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3PharmaceuticalsRetrieval.v2",
    )


class Vidore3PharmaceuticalsRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsRetrieval.v2",
        description="Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PharmaceuticalsOCRRetrieval",
            "revision": "87e635f6d399614218191eb8b9b85ccfd809e6bb",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Medical"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PharmaceuticalsRetrieval"],
    )


class Vidore3ComputerScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_computer_science_mteb_format",
            "revision": "fb7fb69f81f7db62790f40494124b8ad22b424ab",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Programming"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3ComputerScienceRetrieval.v2",
    )


class Vidore3ComputerScienceRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceRetrieval.v2",
        description="Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3ComputerScienceOCRRetrieval",
            "revision": "4d42cf85d8b8117935c04698b5b7fd3c70c8c9f4",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Programming"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3ComputerScienceRetrieval"],
    )


class Vidore3HrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrRetrieval",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_hr_mteb_format",
            "revision": "bc7d43d64815ed30f664168c8052106484aba7fd",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3HrRetrieval.v2",
    )


class Vidore3HrRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrRetrieval.v2",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3HrOCRRetrieval",
            "revision": "dfd4b43eb04bc516860fe5d3cbff67769944833e",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Social"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3HrRetrieval"],
    )


class Vidore3EnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_energy_mteb_format",
            "revision": "84fca99e5978604bae30f2436eacb6dbaa0532e9",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Chemistry", "Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3EnergyRetrieval.v2",
    )


class Vidore3EnergyRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyRetrieval.v2",
        description="Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3EnergyOCRRetrieval",
            "revision": "575fbadfa82b5b5f1d503f4201d1a7d88602b4d8",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Chemistry", "Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3EnergyRetrieval"],
    )


class Vidore3PhysicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "vidore/vidore_v3_physics_mteb_format",
            "revision": "2c18ef90ab3ef93a9d86ecc6521cdae2a29f8300",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3PhysicsRetrieval.v2",
    )


class Vidore3PhysicsRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsRetrieval.v2",
        description="Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PhysicsOCRRetrieval",
            "revision": "948c9fb4d61518c042a59aea5ac3e3d528cea19b",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PhysicsRetrieval"],
    )


class Vidore3NuclearRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3NuclearRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb-private/Vidore3NuclearRetrieval",
            "revision": "a463fc67fefc01152153101e88a32d5f9515e3e3",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Chemistry"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3NuclearRetrieval.v2",
    )


class Vidore3TelecomRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3TelecomRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb-private/Vidore3TelecomRetrieval",
            "revision": "a54635a274ef2835721b7cbe3eb27483b9ec964b",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Programming"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
        contributed_by="Illuin Technology",
        superseded_by="Vidore3TelecomRetrieval.v2",
    )


class Vidore3TelecomRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3TelecomRetrieval.v2",
        description="Retrieve associated pages according to questions."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb-private/Vidore3TelecomOCRRetrieval",
            "revision": "057fe83342b446ede7928cb51bc6c7c3223da3d0",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Programming"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3TelecomRetrieval"],
    )


class Vidore3NuclearRetrievalv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3NuclearRetrieval.v2",
        description="Retrieve associated pages according to questions."
        + "This version add the OCR'ed markdown to allow for comparison across image-text, image-only and text-only models.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb-private/Vidore3NuclearOCRRetrieval",
            "revision": "826e3bc829e01f5a088d0435114200bce6058af8",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Engineering", "Chemistry"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@article{loison2026vidorev3comprehensiveevaluation,
  archiveprefix = {arXiv},
  author = {António Loison and Quentin Macé and Antoine Edy and Victor Xing and Tom Balough and Gabriel Moreira and Bo Liu and Manuel Faysse and Céline Hudelot and Gautier Viaud},
  eprint = {2601.08620},
  primaryclass = {cs.AI},
  title = {ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios},
  url = {https://arxiv.org/abs/2601.08620},
  year = {2026},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3NuclearRetrieval"],
    )
