# RTEB Benchmarks - Retrieval Embedding Benchmark
from __future__ import annotations

from mteb.benchmarks.benchmark import Benchmark
from mteb.overview import get_tasks

RTEB_CITATION = r"""@article{rteb2024,
  author = {RTEB Authors},
  title = {RTEB: Retrieval Embedding Benchmark for Multi-Domain Text Retrieval},
  year = {2024},
}"""

RTEB_MAIN = Benchmark(
    name="RTEB(beta)",
    display_name="RTEB Retrieval Embedding Benchmark",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-search.svg",
    tasks=get_tasks(
        tasks=[
            "AILACasedocs",
            "AILAStatutes",
            "LegalSummarization",
            "LegalQuAD",
            "FinanceBenchRetrieval",
            "HC3FinanceRetrieval",
            "FinQARetrieval",
            "AppsRetrieval",
            "DS1000Retrieval",
            "HumanEvalRetrieval",
            "MBPPRetrieval",
            "WikiSQLRetrieval",
            "FreshStackRetrieval",
            "ChatDoctorRetrieval",
            "CUREv1",
        ],
    ),
    description="RTEB (Retrieval Embedding Benchmark) is a comprehensive benchmark for evaluating text retrieval models across multiple specialized domains including legal, finance, code, and healthcare. It contains 15 diverse retrieval tasks designed to test models' ability to understand domain-specific terminology and retrieve relevant documents in specialized contexts.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_ENGLISH = Benchmark(
    name="RTEB(eng, beta)",
    display_name="RTEB English",
    icon="https://github.com/lipis/flag-icons/raw/refs/heads/main/flags/4x3/us.svg",
    tasks=get_tasks(
        tasks=[
            "AILACasedocs",
            "AILAStatutes",
            "LegalSummarization",
            "FinanceBenchRetrieval",
            "HC3FinanceRetrieval",
            "FinQARetrieval",
            "AppsRetrieval",
            "DS1000Retrieval",
            "HumanEvalRetrieval",
            "MBPPRetrieval",
            "WikiSQLRetrieval",
            "FreshStackRetrieval",
            "ChatDoctorRetrieval",
            "CUREv1",
        ],
        languages=["eng"],
    ),
    description="RTEB English subset containing retrieval tasks in English across legal, finance, code, and healthcare domains.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_FRENCH = Benchmark(
    name="RTEB(fr, beta)",
    display_name="RTEB French",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/fr.svg",
    tasks=get_tasks(
        tasks=[
            "CUREv1",
        ],
        languages=["fra"],
    ),
    description="RTEB French subset containing retrieval tasks in French across multiple domains.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_GERMAN = Benchmark(
    name="RTEB(deu, beta)",
    display_name="RTEB German",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/de.svg",
    tasks=get_tasks(
        tasks=[
            "LegalQuAD",
        ],
    ),
    description="RTEB German subset containing retrieval tasks in German, focusing on legal domain.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_JAPANESE = Benchmark(
    name="RTEB(jpn, beta)",
    display_name="RTEB Japanese",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/jp.svg",
    tasks=get_tasks(
        tasks=[
            # Japanese tasks would go here when available
        ],
    ),
    description="RTEB Japanese subset containing retrieval tasks in Japanese across multiple domains.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_FINANCE = Benchmark(
    name="RTEB(fin, beta)",
    display_name="RTEB Finance",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-price-tag.svg",
    tasks=get_tasks(
        tasks=[
            "FinanceBenchRetrieval",
            "HC3FinanceRetrieval",
            "FinQARetrieval",
        ],
    ),
    description="RTEB Finance subset containing retrieval tasks specifically focused on financial domain including finance benchmarks, Q&A, and financial document retrieval.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_LEGAL = Benchmark(
    name="RTEB(Law, beta)",
    display_name="RTEB Legal",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-library.svg",
    tasks=get_tasks(
        tasks=[
            "AILACasedocs",
            "AILAStatutes",
            "LegalSummarization",
            "LegalQuAD",
        ],
    ),
    description="RTEB Legal subset containing retrieval tasks specifically focused on legal domain including case documents, statutes, legal summarization, and legal Q&A.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_CODE = Benchmark(
    name="RTEB(Code, beta)",
    display_name="RTEB Code",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            "AppsRetrieval",
            "DS1000Retrieval",
            "HumanEvalRetrieval",
            "MBPPRetrieval",
            "WikiSQLRetrieval",
            "FreshStackRetrieval",
        ],
    ),
    description="RTEB Code subset containing retrieval tasks specifically focused on programming and code domains including algorithmic problems, data science tasks, code evaluation, and SQL retrieval.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_HEALTHCARE = Benchmark(
    name="RTEB(Health, beta)",
    display_name="RTEB Healthcare",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-hospital.svg",
    tasks=get_tasks(
        tasks=[
            "ChatDoctorRetrieval",
            "CUREv1",
        ],
    ),
    description="RTEB Healthcare subset containing retrieval tasks specifically focused on healthcare and medical domains including medical Q&A, healthcare information retrieval, and cross-lingual medical retrieval.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)
