"""This sets the outlines for benchmarks to be constructed


The approach is:

1. Define the benchmarks to be constructed using a set of tasks
2. Optionally define tasks that must be included in the benchmark (high quality, representative etc.)
3. Optionally define other limitations such as domains, task type coverage etc. (other conditions such that a task could not be removed)
4. Iteratively remove tasks from the list of tasks based on how predictive they are from the remaining tasks
5. Construct the benchmark using the remaining tasks


This describes the benchmarks that are to be constructed
Along with this I imagine a section called "Build your own benchmark" in the paper where we describe how to construct a benchmark for a domain or language of interest.

Thus we can seek to construct a few large scale benchmarks, but leave the door open for others to construct their own benchmarks.

I believe we have the followign benchmarks to construct:
- MTEB(eng, lite) - a new and faster version of the English MTEB (potentially with a few new tasks? @Niklas?)
- MTEB(Multilingual) - a benchmark that includes all languages (it has to be there)
- A more constrained set of multi-lingual benchmarks:
    - MTEB(European) - a benchmark that includes all European languages, we probably want to include make an extended version of it (including minority languages etc.) - otherwise the story of paper is a bit odd
    - MTEB(African) - a benchmark that includes all African languages (I don't have a lot of experience with this - @imene?)
    - MTEB(Asian) - a benchmark that includes all Asian languages (I don't have a lot of experience with this - @Niklas maybe?)

- Some build-your-own benchmarks dataset for niche cases:
    - MTEB(Scandinavian, v2) - an version of the Mainland Scandinavian benchmark - shows an example of a semi-local language benchmark
    - MTEB(Code) or some other domain specific benchmark - shows an example of a domain specific benchmark
"""

import mteb
from mteb.benchmarks import MTEB_MAIN_EN, MTEB_MAIN_RU

already_established_benchmarks = [
    MTEB_MAIN_RU,  # constructed as a part of MMTEB
]
MMTEB_CITATION = "MISSING"


MMTEB_BENCHMARKS = [
    mteb.Benchmark(
        name="MTEB(eng, lite)",
        tasks=[],  # will be constructed
        description="A lite version of the The English MTEB (Muenninghoff et al., 2023) constructed as a part of MMTEB. The goal is the approximate the score on teh orignal MTEB with a notable reduction in runtime.",
        citation=MMTEB_CITATION,
    )
    # list is unfinished - see below
]


task_lists = {
    "MTEB(eng, lite)": {
        "tasks_to_search": MTEB_MAIN_EN,  # tasks to select from
        "minimally_include": [],  # tasks that must be included in the final benchmark (considered high-quality, representative etc.)
        "task_types_to_include": [  # task types that should be represented in the final benchmark
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
        ],
    },
    "MTEB(Mainland Scandinavian, v2)": {  # intended way to update the benchmarks
        "tasks_to_search": mteb.get_tasks(
            languages=["dan", "nno", "nob", "swe"]
        ),  # and scandinavian tasks
        "minimally_include": [
            # dialect bitext
            "BornholmBitextMining",
            "NorwegianCourtsBitextMining",
            # high quality classification
            "AngryTweetsClassification",
            "DKHateClassification",
            "NoRecClassification",
            "DalajClassification",
            "SweRecClassification",
            # high quality retrieval
            "DanFEVER",
            "NorQuadRetrieval",
            "SweFaqRetrieval",
        ],
        "task_types_to_include": [
            "Classification",
            "Clustering",
            "Reranking",
            "Retrieval",
        ],
    },
    "MTEB(European)": {
        "tasks_to_search": mteb.get_tasks(
            languages=[
                # official EU languages (56) - we could include the whole economic area e.g. Norway - additioanlly we could include minority languages (probably a good idea?)
                # germanic
                "dan",
                "eng",
                "deu",
                "nld",
                "swe",
                # romance
                "fra",
                "ita",
                "por",
                "spa",
                "ron",
                # slavic
                "bul",
                "hrv",
                "ces",
                "pol",
                "slk",
                "slv",
                # baltic
                "lav",
                "lit",
                "est",
                # finno-ugric
                "fin",
                "hun",
                # other indo european
                "ell",
                # non-indo european
                "mlt",
                "gle",
            ]
        ),
        "minimally_include": [],
        # additionally enforce that all language are covered
        "task_types_to_include": [
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
        ],
    },
    "MTEB(African)": {
        "tasks_to_search": mteb.get_tasks(
            languages=[
                # afro asiatic
                "ara",
                "amh",
                "som",
                "tir",
                "hau",
                # Niger Congo
                "swa",
                "yor",
                "ibo",
                "aka",
                "zul",
                "xho",
                "lin",
                # Nilo-Saharan
                "nus",
                "din",
                "kau",
                # Khoisan
                "naq",
                # creole
                "pcm",
            ]
        ),
        "minimally_include": [],
        # additionally enforce that all language are covered
        "task_types_to_include": [
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
        ],
    },
    "MTEB(Asian)": {
        "tasks_to_search": mteb.get_tasks(
            languages=[
                # East asian
                "cmn",
                "yue",
                "jpn",
                "kor",
                "mon"
                # south asian - indic
                # (south asian - dravidian)
                "hin",
                "ben",
                "pan",
                "mar",
                "guj",
                "urd",
                "nep",
                "sin",
                "tam",
                "tel",
                "kan",
                "mal",
                # southeast asian - austronesian
                "ind",
                "msa",
                "fil",
                "jav",
                # southeast asian - tai-kadai
                "tha",
                "lao",
                # southeast asian - austroasiatic
                "vie",
                "khm",
                "mya",
                # central asian - turkic
                "kaz",
                "uzb",
                "tkm",
                "kir",
                "uig",
                # west asian / middle eastern - semitic
                "ara",
                "heb",
                # west asian / middle eastern - iranian
                "fas",
                "kur",
                "pus",
                "prs",
            ]
        ),
        "minimally_include": [],
        # additionally enforce that all language are covered
        "task_types_to_include": [
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
        ],
    },
    "MTEB(Multilingual)": {
        "tasks_to_search": mteb.get_tasks(),  # just use everything
        "minimally_include": [
            "BibleNLPBitextMining",  # extreme coverage
            "SIB200ClusteringS2S",
            "MultilingualSentimentClassification",
            "MultiEURLEXMultilabelClassification",
            "MasakhaNEWSClassification",
            "MIRACLRetrieval",
            "MIRACLReranking",
            "WikiClusteringP2P.v2",
            "XQuADRetrieval",
            "SemRel24STS",
        ],
        "task_types_to_include": [
            "Classification",
            "Clustering",
            "PairClassification",
            "Reranking",
            "Retrieval",
        ],
    },
}
