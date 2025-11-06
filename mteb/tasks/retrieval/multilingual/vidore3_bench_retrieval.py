from datasets import load_dataset

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


def _load_data(
    path: str,
    splits: list[str],
    langs: list | None = None,
    revision: str | None = None,
):
    query_columns_to_remove = [
        "query_id",
        "query",
        "query_types",
        "query_format",
        "content_type",
        "raw_answers",
        "raw_answers",
        "query_generator",
        "query_generation_pipeline",
        "source_type",
        "query_type_for_generation",
        "answer",
    ]

    qrel_columns_to_remove = ["content_type", "bounding_boxes"]

    corpus_columns_to_remove = [
        "corpus_id",
        "doc_id",
        "markdown",
        "page_number_in_doc",
    ]

    if langs is None:
        corpus = {}
        queries = {}
        relevant_docs = {}
    else:
        corpus = {lang: {} for lang in langs}
        queries = {lang: {} for lang in langs}
        relevant_docs = {lang: {} for lang in langs}

    for split in splits:
        query_ds = load_dataset(
            path,
            "queries",
            split=split,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['query_id']}",
                "text": x["query"],
            },
            remove_columns=query_columns_to_remove,
        )

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['corpus_id']}",
            },
            remove_columns=corpus_columns_to_remove,
        )

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            revision=revision,
        )
        qrels_ds = qrels_ds.remove_columns(qrel_columns_to_remove)

        if langs is None:
            queries[split] = query_ds
            corpus[split] = corpus_ds
            relevant_docs[split] = {}
            for row in qrels_ds:
                qid = f"query-{split}-{row['query_id']}"
                did = f"corpus-{split}-{row['corpus_id']}"
                if qid not in relevant_docs[split]:
                    relevant_docs[split][qid] = {}
                relevant_docs[split][qid][did] = int(row["score"])
        else:
            for lang in langs:
                queries[lang][split] = query_ds.filter(lambda x: x["language"] == lang)

                corpus[lang][split] = corpus_ds

                relevant_docs[lang][split] = {}
                for row in qrels_ds:
                    qid = f"query-{split}-{row['query_id']}"
                    did = f"corpus-{split}-{row['corpus_id']}"
                    if qid not in relevant_docs[lang][split]:
                        relevant_docs[lang][split][qid] = {}
                    relevant_docs[lang][split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs


class Vidore3FinanceEnRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - EN, is a corpus of reports from american banking companies, intended for long-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )

class Vidore3FinanceFrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrRetrieval",
        description="Retrieve associated pages according to questions. This task, Finance - FR, is a corpus of reports from french companies in the luxury domain, intended for long-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3IndustrialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Industrial reports, is a corpus of technical documents on military aircrafts (fueling, mechanics...), intended for complex-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3PharmaceuticalsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Pharmaceutical, is a corpus of slides from the FDA, intended for long-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )

class Vidore3ComputerScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Computer Science, is a corpus of textbooks from the openstacks website, intended for long-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3HrRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrRetrieval",
        description="Retrieve associated pages according to questions. This dataset, HR, is a corpus of reports released by the european union, intended for complex-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3EnergyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Energy Fr, is a corpus of reports on energy supply in europe, intended for complex-document understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3PhysicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsRetrieval",
        description="Retrieve associated pages according to questions. This dataset, Physics, is a corpus of course slides on bachelor level physics lectures, intended for complex visual understanding tasks.",
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
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=True,
    )


class Vidore3NuclearRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3NuclearRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
        dataset={
            "path": "mysecretorga/energy_nuclear_plant_en",
            "revision": "ce6c1e966eb26d094ce206897dd87452fed1fdaa",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True


class Vidore3TelecomRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3TelecomRetrieval",
        description="Retrieve associated pages according to questions.",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
        dataset={
            "path": "mysecretorga/telecom_internet_protocols",
            "revision": "1dbb47240bd521108cf82b58a885555cce4c5612",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        date=("2025-10-01", "2025-11-01"),
        domains=["Academic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
}""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
        is_public=False,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            langs=["english", "french", "spanish", "german", "italian", "portuguese"],
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
