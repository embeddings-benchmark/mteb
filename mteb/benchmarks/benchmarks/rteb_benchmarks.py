# RTEB Benchmarks - Retrieval Embedding Benchmark


from mteb.benchmarks.benchmark import RtebBenchmark
from mteb.get_tasks import get_tasks

RTEB_CITATION = r"""@article{rteb2025,
  author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
  title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
  year = {2025},
}"""

RTEB_MAIN = RtebBenchmark(
    name="RTEB(beta)",
    display_name="RTEB Multilingual",
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
            "MIRACLRetrievalHardNegatives",
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
    description="RTEB (ReTrieval Embedding Benchmark) is a comprehensive benchmark for evaluating text retrieval models across multiple specialized domains including legal, finance, code, and healthcare. It contains diverse retrieval tasks designed to test models' ability to understand domain-specific terminology and retrieve relevant documents in specialized contexts across multiple languages. The dataset includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_ENGLISH = RtebBenchmark(
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
    description="RTEB English is a subset of RTEB containing retrieval tasks in English across legal, finance, code, and healthcare domains. Includes diverse tasks covering specialized domains such as healthcare and finance. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_FRENCH = RtebBenchmark(
    name="RTEB(fra, beta)",
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
    description="RTEB French is a subset of RTEB containing retrieval tasks in French across legal and general knowledge domains. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_GERMAN = RtebBenchmark(
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
    description="RTEB German is a subset of RTEB containing retrieval tasks in German across legal, healthcare, and business domains. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_JAPANESE = RtebBenchmark(
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
    description="RTEB Japanese is a subset of RTEB  containing retrieval tasks in Japanese across legal and code domains. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_FINANCE = RtebBenchmark(
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
    description="RTEB Finance is a subset of RTEB  containing retrieval tasks specifically focused on financial domain including finance benchmarks, Q&A, financial document retrieval, and corporate governance. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_LEGAL = RtebBenchmark(
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
    description="RTEB Legal is a subset of RTEB containing retrieval tasks specifically focused on legal domain including case documents, statutes, legal summarization, and multilingual legal Q&A. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_CODE = RtebBenchmark(
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
    description="RTEB Code is a subset of RTEB containing retrieval tasks specifically focused on programming and code domains including algorithmic problems, data science tasks, code evaluation, SQL retrieval, and multilingual code retrieval. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_HEALTHCARE = RtebBenchmark(
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
    description="RTEB Healthcare is a subset of RTEB containing retrieval tasks specifically focused on healthcare and medical domains including medical Q&A, healthcare information retrieval, cross-lingual medical retrieval, and multilingual medical consultation. The benchmark includes both open and closed datasets, providing a robust evaluation framework for real-world applications. To submit results on private tasks, please create [open an issue](https://github.com/embeddings-benchmark/mteb/issues).",
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)
