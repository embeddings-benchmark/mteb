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
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )


class Vidore3FinanceFrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
        sample_creation="created and machine-translated",
        bibtex_citation=r"""
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3IndustrialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircraft (fueling, mechanics...), intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3PharmaceuticalsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3ComputerScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3HrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrRetrieval",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks. Original queries were created in english, then translated to french, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3EnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3PhysicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on french bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in french, then translated to english, german, italian, portuguese and spanish.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3NuclearRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3NuclearRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
    )


class Vidore3TelecomRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3TelecomRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
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
@misc{mace2025vidorev3,
  author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  day = {5},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
  journal = {Hugging Face Blog},
  month = {November},
  publisher = {Hugging Face},
  title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
    )
