# RTEB Benchmarks - Retrieval Embedding Benchmark


from mteb.benchmarks.benchmark import RtebBenchmark
from mteb.get_tasks import MTEBTasks, get_task, get_tasks

RTEB_CITATION = r"""@article{rteb2025,
  author = {Liu, Frank and Enevoldsen, Kenneth and Solomatin, Roman and Chung, Isaac and Aarsen, Tom and Fődi, Zoltán},
  title = {Introducing RTEB: A New Standard for Retrieval Evaluation},
  year = {2025},
}"""

removal_note = "\n\nNote: We have temporarily removed the 'Private' column to read more about this decision out the [announcement](https://github.com/embeddings-benchmark/mteb/issues/3934)."

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
            "SWEbenchCodeRetrieval",
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
    description="Retrieval quality across specialized domains including legal, finance, code, and healthcare in multiple languages, with tasks representative of real-world production retrieval demands. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)

RTEB_ENGLISH = RtebBenchmark(
    name="RTEB(eng, beta)",
    display_name="RTEB English",
    icon="https://github.com/lipis/flag-icons/raw/refs/heads/main/flags/4x3/us.svg",
    tasks=MTEBTasks(
        get_tasks(
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
                "SWEbenchCodeRetrieval",
                "ChatDoctorRetrieval",
                # Closed datasets
                "Code1Retrieval",
                "EnglishFinance1Retrieval",
                "EnglishFinance2Retrieval",
                "EnglishFinance3Retrieval",
                "EnglishFinance4Retrieval",
                "EnglishHealthcare1Retrieval",
            ],
            languages=["eng"],
        )
        + (
            get_task(
                "CUREv1",
                hf_subsets="en",
            ),
        )
    ),
    description="Retrieval quality in English across legal, finance, code, and healthcare domains, with tasks representative of real-world production retrieval demands. An English-only subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
    description="Retrieval quality in French across legal and general knowledge domains, with tasks representative of real-world production retrieval demands. A French-language subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
    description="Retrieval quality in German across legal, healthcare, and business domains, with tasks representative of real-world production retrieval demands. A German-language subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
    description="Retrieval quality in Japanese across legal and code domains, with tasks representative of real-world production retrieval demands. A Japanese-language subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
    description="Retrieval quality in the financial domain across finance benchmarks, Q&A, financial document retrieval, and corporate governance, with tasks representative of real-world production retrieval demands. A domain-specific subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
    description="Retrieval quality in the legal domain across case documents, statutes, legal summarization, and multilingual legal Q&A, with tasks representative of real-world production retrieval demands. A domain-specific subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
            "SWEbenchCodeRetrieval",
            # Closed datasets
            "Code1Retrieval",
            "JapaneseCode1Retrieval",
        ],
    ),
    description="Retrieval quality in the code domain across algorithmic problems, data science tasks, code evaluation, SQL retrieval, and multilingual code retrieval, with tasks representative of real-world production retrieval demands. A domain-specific subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
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
    description="Retrieval quality in the healthcare and medical domain across medical Q&A, healthcare information retrieval, and multilingual medical consultation, with tasks representative of real-world production retrieval demands. A domain-specific subset of RTEB. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues)."
    + removal_note,
    citation=RTEB_CITATION,
    contacts=["fzowl"],
)
