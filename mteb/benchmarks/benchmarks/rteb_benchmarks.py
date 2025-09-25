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
            # Closed datasets
            "Code1Retrieval",
            "JapaneseCode1Retrieval",
            "EnglishFinance1Retrieval",
            "EnglishFinance2Retrieval",
            "EnglishFinance3Retrieval",
            "EnglishFinance4Retrieval",
            "EnglishHealthcare1Retrieval",
            "French1Retrieval",
            "FrenchLegal1Retrieval",
            "German1Retrieval",
            "GermanHealthcare1Retrieval",
            "GermanLegal1Retrieval",
            "JapaneseLegal1Retrieval",
        ],
    ),
    description="RTEB (Retrieval Embedding Benchmark) is a comprehensive benchmark for evaluating text retrieval models across multiple specialized domains including legal, finance, code, and healthcare. It contains 28 diverse retrieval tasks designed to test models' ability to understand domain-specific terminology and retrieve relevant documents in specialized contexts across English, French, German, and Japanese languages.",
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
            # Closed datasets
            "Code1Retrieval",
            "EnglishFinance1Retrieval",
            "EnglishFinance2Retrieval",
            "EnglishFinance3Retrieval",
            "EnglishFinance4Retrieval",
            "EnglishHealthcare1Retrieval",
        ],
        languages=["eng"],
    ),
    description="RTEB English subset containing retrieval tasks in English across legal, finance, code, and healthcare domains. Includes 20 diverse tasks covering specialized domains.",
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
            # Closed datasets
            "French1Retrieval",
            "FrenchLegal1Retrieval",
        ],
        languages=["fra"],
    ),
    description="RTEB French subset containing retrieval tasks in French across legal and general knowledge domains. Includes 3 diverse multilingual tasks.",
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
            # Closed datasets
            "German1Retrieval",
            "GermanHealthcare1Retrieval",
            "GermanLegal1Retrieval",
        ],
    ),
    description="RTEB German subset containing retrieval tasks in German across legal, healthcare, and business domains. Includes 4 diverse tasks.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_JAPANESE = Benchmark(
    name="RTEB(jpn, beta)",
    display_name="RTEB Japanese",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/jp.svg",
    tasks=get_tasks(
        tasks=[
            # Closed datasets
            "JapaneseCode1Retrieval",
            "JapaneseLegal1Retrieval",
        ],
    ),
    description="RTEB Japanese subset containing retrieval tasks in Japanese across legal and code domains. Includes 2 diverse multilingual tasks.",
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
            # Closed datasets
            "EnglishFinance1Retrieval",
            "EnglishFinance2Retrieval",
            "EnglishFinance3Retrieval",
            "EnglishFinance4Retrieval",
        ],
    ),
    description="RTEB Finance subset containing retrieval tasks specifically focused on financial domain including finance benchmarks, Q&A, financial document retrieval, and corporate governance. Includes 7 specialized finance tasks.",
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
            # Closed datasets
            "FrenchLegal1Retrieval",
            "GermanLegal1Retrieval",
            "JapaneseLegal1Retrieval",
        ],
    ),
    description="RTEB Legal subset containing retrieval tasks specifically focused on legal domain including case documents, statutes, legal summarization, and multilingual legal Q&A. Includes 7 legal tasks across English, French, German, and Japanese.",
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
            # Closed datasets
            "Code1Retrieval",
            "JapaneseCode1Retrieval",
        ],
    ),
    description="RTEB Code subset containing retrieval tasks specifically focused on programming and code domains including algorithmic problems, data science tasks, code evaluation, SQL retrieval, and multilingual code retrieval. Includes 8 code-related tasks.",
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
            # Closed datasets
            "EnglishHealthcare1Retrieval",
            "GermanHealthcare1Retrieval",
        ],
    ),
    description="RTEB Healthcare subset containing retrieval tasks specifically focused on healthcare and medical domains including medical Q&A, healthcare information retrieval, cross-lingual medical retrieval, and multilingual medical consultation. Includes 4 healthcare tasks.",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)
