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


class Vidore3ComputerScienceBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceBGEm3Reranking.v2",
        description="Reranking task for retrieval on Computer Science domain. The corpus is composed by textbooks from the OpenStax website, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3ComputerScienceBGEm3Reranking.v2",
            "revision": "16e4156dc05941dc98b48de38a5ab68732a963c5",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3ComputerScienceRetrieval.v2"],
    )


class Vidore3ComputerScienceBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceBM25sReranking.v2",
        description="Reranking task for retrieval on Computer Science domain. The corpus is composed by textbooks from the OpenStax website, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3ComputerScienceBM25sReranking.v2",
            "revision": "7ea0214224f4893c6366c98da2a233e8049d75d8",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3ComputerScienceRetrieval.v2"],
    )


class Vidore3ComputerScienceQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3ComputerScienceQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Computer Science domain. The corpus is composed by textbooks from the OpenStax website, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3ComputerScienceQwen3VLEmbedding2BReranking.v2",
            "revision": "2a0ac5319d2c280bb315b4723baed216e71f2766",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3ComputerScienceRetrieval.v2"],
    )


class Vidore3EnergyBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyBGEm3Reranking.v2",
        description="Reranking task for retrieval on Energy domain. The corpus is composed by reports on energy supply in Europe, intended for complex-document understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3EnergyBGEm3Reranking.v2",
            "revision": "39d566f868623dacd6a7529d60eb68f2b1f1580f",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3EnergyRetrieval.v2"],
    )


class Vidore3EnergyBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyBM25sReranking.v2",
        description="Reranking task for retrieval on Energy domain. The corpus is composed by reports on energy supply in Europe, intended for complex-document understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3EnergyBM25sReranking.v2",
            "revision": "75b03f07ac6fbef803a44c8fff3c1215a22ad85f",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3EnergyRetrieval.v2"],
    )


class Vidore3EnergyQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3EnergyQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Energy domain. The corpus is composed by reports on energy supply in Europe, intended for complex-document understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3EnergyQwen3VLEmbedding2BReranking.v2",
            "revision": "4964d8567d0428feec131ef5b3e6ab37c8f80599",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3EnergyRetrieval.v2"],
    )


class Vidore3FinanceEnBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnBGEm3Reranking.v2",
        description="Reranking task for retrieval on Finance - EN domain. The corpus is composed by reports from American banking companies, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceEnBGEm3Reranking.v2",
            "revision": "cbfb6aa95395765d61dced7ec061b7f5ffb0ca10",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        adapted_from=["Vidore3FinanceEnRetrieval.v2"],
        contributed_by="Illuin Technology",
    )


class Vidore3FinanceEnBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnBM25sReranking.v2",
        description="Reranking task for retrieval on Finance - EN domain. The corpus is composed by reports from American banking companies, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceEnBM25sReranking.v2",
            "revision": "7187c21e1a532531d175eccf5a6d7c1561f6b0da",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        adapted_from=["Vidore3FinanceEnRetrieval.v2"],
        contributed_by="Illuin Technology",
    )


class Vidore3FinanceEnQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceEnQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Finance - EN domain. The corpus is composed by reports from American banking companies, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceEnQwen3VLEmbedding2BReranking.v2",
            "revision": "670019b512b99991bcca43e8ff841c7065c682c6",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        adapted_from=["Vidore3FinanceEnRetrieval.v2"],
        contributed_by="Illuin Technology",
    )


class Vidore3FinanceFrBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrBGEm3Reranking.v2",
        description="Reranking task for retrieval on Finance - FR domain. The corpus is composed by reports from French companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceFrBGEm3Reranking.v2",
            "revision": "8e96fdacf2d8bd48fc69d8664244a10ef53562f4",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3FinanceFrRetrieval.v2"],
    )


class Vidore3FinanceFrBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrBM25sReranking.v2",
        description="Reranking task for retrieval on Finance - FR domain. The corpus is composed by reports from French companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceFrBM25sReranking.v2",
            "revision": "5c4b3ef4b996bc23625c931d32fd9e7e2b3066af",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3FinanceFrRetrieval.v2"],
    )


class Vidore3FinanceFrQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3FinanceFrQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Finance - FR domain. The corpus is composed by reports from French companies in the luxury domain, intended for long-document understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3FinanceFrQwen3VLEmbedding2BReranking.v2",
            "revision": "a439545a55d4bc8082b4d8a14e09e277b72c9751",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3FinanceFrRetrieval.v2"],
    )


class Vidore3HrBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrBGEm3Reranking.v2",
        description="Reranking task for retrieval on HR domain. The corpus is composed by reports released by the European Union, intended for complex-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3HrBGEm3Reranking.v2",
            "revision": "c4c411c26f9b7b92966001538a919c6395ed7149",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3HrRetrieval.v2"],
    )


class Vidore3HrBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrBM25sReranking.v2",
        description="Reranking task for retrieval on HR domain. The corpus is composed by reports released by the European Union, intended for complex-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3HrBM25sReranking.v2",
            "revision": "25dd9c87572accc3488d5b349c0760c8b2d77dfb",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3HrRetrieval.v2"],
    )


class Vidore3HrQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3HrQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on HR domain. The corpus is composed by reports released by the European Union, intended for complex-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3HrQwen3VLEmbedding2BReranking.v2",
            "revision": "c41fbd8791399ce8668635c3618decebcd328be7",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3HrRetrieval.v2"],
    )


class Vidore3IndustrialBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialBGEm3Reranking.v2",
        description="Reranking task for retrieval on Industrial Reports domain. The corpus is composed by technical documents on military aircraft (fueling, mechanics, etc.), intended for complex-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3IndustrialBGEm3Reranking.v2",
            "revision": "becf4ff4dd02c70c774730aed26be956171a7ad8",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3IndustrialRetrieval.v2"],
    )


class Vidore3IndustrialBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialBM25sReranking.v2",
        description="Reranking task for retrieval on Industrial Reports domain. The corpus is composed by technical documents on military aircraft (fueling, mechanics, etc.), intended for complex-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3IndustrialBM25sReranking.v2",
            "revision": "feb084441e6fc28580a023479211715c28700653",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3IndustrialRetrieval.v2"],
    )


class Vidore3IndustrialQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3IndustrialQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Industrial Reports domain. The corpus is composed by technical documents on military aircraft (fueling, mechanics, etc.), intended for complex-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3IndustrialQwen3VLEmbedding2BReranking.v2",
            "revision": "11b26df1c350556c5341d229a5f0830e14dab95e",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3IndustrialRetrieval.v2"],
    )


class Vidore3PharmaceuticalsBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsBGEm3Reranking.v2",
        description="Reranking task for retrieval on Pharmaceutical domain. The corpus is composed by slides from the FDA, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PharmaceuticalsBGEm3Reranking.v2",
            "revision": "f2561cf9ba64223f3c72168eeb1ab41a70749be4",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PharmaceuticalsRetrieval.v2"],
    )


class Vidore3PharmaceuticalsBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsBM25sReranking.v2",
        description="Reranking task for retrieval on Pharmaceutical domain. The corpus is composed by slides from the FDA, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PharmaceuticalsBM25sReranking.v2",
            "revision": "6d20dfc6799e48a4b1d9dab909980c06b9caa3d3",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PharmaceuticalsRetrieval.v2"],
    )


class Vidore3PharmaceuticalsQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PharmaceuticalsQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Pharmaceutical domain. The corpus is composed by slides from the FDA, intended for long-document understanding tasks. Original queries were created in English, then translated to French, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PharmaceuticalsQwen3VLEmbedding2BReranking.v2",
            "revision": "1ebaef42ac7f10a35300a66dee9f3bf570575264",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PharmaceuticalsRetrieval.v2"],
    )


class Vidore3PhysicsBGEm3Rerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsBGEm3Reranking.v2",
        description="Reranking task for retrieval on Physics domain. The corpus is composed by course slides on French bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the BAAI/bge-m3 model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PhysicsBGEm3Reranking.v2",
            "revision": "8152b6f8843f7d5471804e6b8b998d9114bd67e1",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PhysicsRetrieval.v2"],
    )


class Vidore3PhysicsBM25sRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsBM25sReranking.v2",
        description="Reranking task for retrieval on Physics domain. The corpus is composed by course slides on French bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the mteb/baseline-bm25s model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PhysicsBM25sReranking.v2",
            "revision": "6e5866da01375cd114ccd28c3c77e1ba795392c7",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PhysicsRetrieval.v2"],
    )


class Vidore3PhysicsQwen3VLEmbedding2BRerankingv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Vidore3PhysicsQwen3VLEmbedding2BReranking.v2",
        description="Reranking task for retrieval on Physics domain. The corpus is composed by course slides on French bachelor level physics lectures, intended for complex visual understanding tasks. Original queries were created in French, then translated to English, German, Italian, Portuguese and Spanish. The candidates for reranking are the top-50 pages retrieved by the Qwen/Qwen3-VL-Embedding-2B model.",
        reference="https://arxiv.org/abs/2601.08620",
        dataset={
            "path": "mteb/Vidore3PhysicsQwen3VLEmbedding2BReranking.v2",
            "revision": "8db58924b1280eb29deaf4af7b6ceb32767a91c5",
        },
        type="Reranking",
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
        prompt={"query": "Retrieve images or text relevant to the user's query."},
        is_public=True,
        contributed_by="Illuin Technology",
        adapted_from=["Vidore3PhysicsRetrieval.v2"],
    )
