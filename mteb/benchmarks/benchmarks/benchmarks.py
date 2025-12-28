from mteb.benchmarks.benchmark import (
    Benchmark,
    HUMEBenchmark,
    MIEBBenchmark,
    VidoreBenchmark,
)
from mteb.get_tasks import MTEBTasks, get_task, get_tasks

MMTEB_CITATION = r"""@article{enevoldsen2025mmtebmassivemultilingualtext,
  author = {Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  doi = {10.48550/arXiv.2502.13595},
  journal = {arXiv preprint arXiv:2502.13595},
  publisher = {arXiv},
  title = {MMTEB: Massive Multilingual Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2502.13595},
  year = {2025},
}"""

MTEB_EN = Benchmark(
    name="MTEB(eng, v2)",
    aliases=["MTEB(eng)"],
    display_name="English",
    icon="https://github.com/lipis/flag-icons/raw/refs/heads/main/flags/4x3/us.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "ArguAna",
                "ArXivHierarchicalClusteringP2P",
                "ArXivHierarchicalClusteringS2S",
                "AskUbuntuDupQuestions",
                "BIOSSES",
                "Banking77Classification",
                "BiorxivClusteringP2P.v2",
                "CQADupstackGamingRetrieval",
                "CQADupstackUnixRetrieval",
                "ClimateFEVERHardNegatives",
                "FEVERHardNegatives",
                "FiQA2018",
                "HotpotQAHardNegatives",
                "ImdbClassification",
                "MTOPDomainClassification",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "MedrxivClusteringP2P.v2",
                "MedrxivClusteringS2S.v2",
                "MindSmallReranking",
                "SCIDOCS",
                "SICK-R",
                "STS12",
                "STS13",
                "STS14",
                "STS15",
                "STSBenchmark",
                "SprintDuplicateQuestions",
                "StackExchangeClustering.v2",
                "StackExchangeClusteringP2P.v2",
                "TRECCOVID",
                "Touche2020Retrieval.v3",
                "ToxicConversationsClassification",
                "TweetSentimentExtractionClassification",
                "TwentyNewsgroupsClustering.v2",
                "TwitterSemEval2015",
                "TwitterURLCorpus",
                "SummEvalSummarization.v2",
            ],
            languages=["eng"],
            eval_splits=["test"],
            exclusive_language_filter=True,
        )
        + (
            get_task(
                "AmazonCounterfactualClassification",
                eval_splits=["test"],
                hf_subsets=["en"],
            ),
            get_task("STS17", eval_splits=["test"], hf_subsets=["en-en"]),
            get_task("STS22.v2", eval_splits=["test"], hf_subsets=["en"]),
        ),
    ),
    description="""The new English Massive Text Embedding Benchmark.
This benchmark was created to account for the fact that many models have now been finetuned
to tasks in the original MTEB, and contains tasks that are not as frequently used for model training.
This way the new benchmark and leaderboard can give our users a more realistic expectation of models' generalization performance.

The original MTEB leaderboard is available under the [MTEB(eng, v1)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v1%29) tab.
    """,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "Muennighoff"],
)

MTEB_ENG_CLASSIC = Benchmark(
    name="MTEB(eng, v1)",
    aliases=["MTEB(eng, classic)", "MTEB"],
    display_name="English Legacy",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/gb.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "AmazonPolarityClassification",
                "AmazonReviewsClassification",
                "ArguAna",
                "ArxivClusteringP2P",
                "ArxivClusteringS2S",
                "AskUbuntuDupQuestions",
                "BIOSSES",
                "Banking77Classification",
                "BiorxivClusteringP2P",
                "BiorxivClusteringS2S",
                "CQADupstackRetrieval",
                "ClimateFEVER",
                "DBPedia",
                "EmotionClassification",
                "FEVER",
                "FiQA2018",
                "HotpotQA",
                "ImdbClassification",
                "MTOPDomainClassification",
                "MTOPIntentClassification",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "MedrxivClusteringP2P",
                "MedrxivClusteringS2S",
                "MindSmallReranking",
                "NFCorpus",
                "NQ",
                "QuoraRetrieval",
                "RedditClustering",
                "RedditClusteringP2P",
                "SCIDOCS",
                "SICK-R",
                "STS12",
                "STS13",
                "STS14",
                "STS15",
                "STS16",
                "STSBenchmark",
                "SciDocsRR",
                "SciFact",
                "SprintDuplicateQuestions",
                "StackExchangeClustering",
                "StackExchangeClusteringP2P",
                "StackOverflowDupQuestions",
                "SummEval",
                "TRECCOVID",
                "Touche2020",
                "ToxicConversationsClassification",
                "TweetSentimentExtractionClassification",
                "TwentyNewsgroupsClustering",
                "TwitterSemEval2015",
                "TwitterURLCorpus",
            ],
            languages=["eng"],
            eval_splits=["test"],
        )
        + get_tasks(tasks=["MSMARCO"], languages=["eng"], eval_splits=["dev"])
        + (
            get_task(
                "AmazonCounterfactualClassification",
                eval_splits=["test"],
                hf_subsets=["en"],
            ),
            get_task(
                "STS17",
                eval_splits=["test"],
                hf_subsets=["en-en"],
            ),
            get_task("STS22", eval_splits=["test"], hf_subsets=["en"]),
        )
    ),
    description="""The original English benchmark by Muennighoff et al., (2023).
This page is an adaptation of the [old MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard_legacy).
We recommend that you use [MTEB(eng, v2)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v2%29) instead, as it uses updated versions of the task, making it notably faster to run and resolving [a known bug](https://github.com/embeddings-benchmark/mteb/issues/1156) in existing tasks. This benchmark also removes datasets common for fine-tuning, such as MSMARCO, which makes model performance scores more comparable. However, generally, both benchmarks provide similar estimates.
    """,
    citation=r"""
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  doi = {10.48550/ARXIV.2210.07316},
  journal = {arXiv preprint arXiv:2210.07316},
  publisher = {arXiv},
  title = {MTEB: Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2210.07316},
  year = {2022},
}
""",
    contacts=["Muennighoff"],
)

MTEB_MAIN_RU = Benchmark(
    name="MTEB(rus, v1)",
    aliases=["MTEB(rus)"],
    display_name="Russian legacy",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/ru.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["rus"],
            tasks=[
                # Classification
                "GeoreviewClassification",
                "HeadlineClassification",
                "InappropriatenessClassification",
                "KinopoiskClassification",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "RuReviewsClassification",
                "RuSciBenchGRNTIClassification",
                "RuSciBenchOECDClassification",
                # Clustering
                "GeoreviewClusteringP2P",
                "RuSciBenchGRNTIClusteringP2P",
                "RuSciBenchOECDClusteringP2P",
                # MultiLabelClassification
                "CEDRClassification",
                "SensitiveTopicsClassification",
                # PairClassification
                "TERRa",
                # Reranking
                "MIRACLReranking",
                "RuBQReranking",
                # Retrieval
                "MIRACLRetrieval",
                "RiaNewsRetrieval",
                "RuBQRetrieval",
                # STS
                "RUParaPhraserSTS",
                "STS22",
            ],
        )
        + get_tasks(
            tasks=["RuSTSBenchmarkSTS"],
            eval_splits=["test"],
        )
    ),
    description="A Russian version of the Massive Text Embedding Benchmark with a number of novel Russian tasks in all task categories of the original MTEB.",
    reference="https://aclanthology.org/2023.eacl-main.148/",
    citation=r"""
@misc{snegirev2024russianfocusedembeddersexplorationrumteb,
  archiveprefix = {arXiv},
  author = {Artem Snegirev and Maria Tikhonova and Anna Maksimova and Alena Fenogenova and Alexander Abramov},
  eprint = {2408.12503},
  primaryclass = {cs.CL},
  title = {The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design},
  url = {https://arxiv.org/abs/2408.12503},
  year = {2024},
}
""",
    contacts=["Samoed", "artemsnegirev", "Drozhzhinastya"],
)

MTEB_MAIN_RU_v1_1 = Benchmark(
    name="MTEB(rus, v1.1)",
    display_name="Russian",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/ru.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["rus"],
            tasks=[
                # Classification
                "GeoreviewClassification",
                "HeadlineClassification",
                "InappropriatenessClassification",
                "KinopoiskClassification",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "RuReviewsClassification",
                "RuSciBenchGRNTIClassification",
                "RuSciBenchOECDClassification",
                # Clustering
                "GeoreviewClusteringP2P",
                "RuSciBenchGRNTIClusteringP2P",
                "RuSciBenchOECDClusteringP2P",
                # MultiLabelClassification
                "CEDRClassification",
                "SensitiveTopicsClassification",
                # PairClassification
                "TERRa",
                # Reranking
                "MIRACLReranking",
                "RuBQReranking",
                # Retrieval
                "MIRACLRetrievalHardNegatives.v2",
                "RiaNewsRetrievalHardNegatives.v2",
                "RuBQRetrieval",
                # STS
                "RUParaPhraserSTS",
                "STS22",
            ],
        )
        + get_tasks(
            tasks=["RuSTSBenchmarkSTS"],
            eval_splits=["test"],
        )
    ),
    description="A Russian version of the Massive Text Embedding Benchmark covering the task categories of classification, clustering, reranking, pair classification, retrieval, and semantic similarity. In v1.1, MIRACLRetrieval and RiaNewsRetrieval were replaced with their HardNegatives variants for improved time-optimization measurement. MIRACLRetrievalHardNegatives and RiaNewsRetrievalHardNegatives are used in their updated versions (v2), both of which include improved default prompts.",
    reference="https://aclanthology.org/2023.eacl-main.148/",
    citation=r"""
@misc{snegirev2024russianfocusedembeddersexplorationrumteb,
  archiveprefix = {arXiv},
  author = {Artem Snegirev and Maria Tikhonova and Anna Maksimova and Alena Fenogenova and Alexander Abramov},
  eprint = {2408.12503},
  primaryclass = {cs.CL},
  title = {The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design},
  url = {https://arxiv.org/abs/2408.12503},
  year = {2024},
}
""",
    contacts=["Samoed", "artemsnegirev", "Drozhzhinastya"],
)


RU_SCI_BENCH = Benchmark(
    name="RuSciBench",
    tasks=get_tasks(
        tasks=[
            # BitextMining
            "RuSciBenchBitextMining.v2",
            # Classification
            "RuSciBenchCoreRiscClassification",
            "RuSciBenchGRNTIClassification.v2",
            "RuSciBenchOECDClassification.v2",
            "RuSciBenchPubTypeClassification",
            # Retrieval
            "RuSciBenchCiteRetrieval",
            "RuSciBenchCociteRetrieval",
            # Regression
            "RuSciBenchCitedCountRegression",
            "RuSciBenchYearPublRegression",
        ],
    ),
    description="RuSciBench is a benchmark designed for evaluating sentence encoders and language models on scientific texts in both Russian and English. The data is sourced from eLibrary (www.elibrary.ru), Russia's largest electronic library of scientific publications. This benchmark facilitates the evaluation and comparison of models on various research-related tasks.",
    reference="https://link.springer.com/article/10.1134/S1064562424602191",
    citation=r"""
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
)

MTEB_RETRIEVAL_WITH_INSTRUCTIONS = Benchmark(
    name="FollowIR",
    aliases=["MTEB(Retrieval w/Instructions)"],
    display_name="Instruction Following",
    tasks=get_tasks(
        tasks=[
            "Robust04InstructionRetrieval",
            "News21InstructionRetrieval",
            "Core17InstructionRetrieval",
        ]
    ),
    description="Retrieval w/Instructions is the task of finding relevant documents for a query that has detailed instructions.",
    reference="https://arxiv.org/abs/2403.15246",
    citation=r"""
@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
}
""",
)

MTEB_RETRIEVAL_WITH_DOMAIN_INSTRUCTIONS = Benchmark(
    name="IFIR",
    display_name="IFIR",
    tasks=get_tasks(
        tasks=[
            "IFIRAila",
            "IFIRCds",
            "IFIRFiQA",
            "IFIRFire",
            "IFIRNFCorpus",
            "IFIRPm",
            "IFIRScifact",
        ]
    ),
    description="Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval",
    reference="https://arxiv.org/abs/2503.04644",
    citation=r"""
@inproceedings{song2025ifir,
  author = {Song, Tingyu and Gan, Guo and Shang, Mingsheng and Zhao, Yilun},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages = {10186--10204},
  title = {IFIR: A Comprehensive Benchmark for Evaluating Instruction-Following in Expert-Domain Information Retrieval},
  year = {2025},
}
""",
)

MTEB_RETRIEVAL_LAW = Benchmark(
    # This benchmark is likely in the need of an update
    name="MTEB(Law, v1)",
    aliases=["MTEB(law)"],
    display_name="Legal",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-library.svg",
    tasks=get_tasks(
        tasks=[
            "AILACasedocs",
            "AILAStatutes",
            "LegalSummarization",
            "GerDaLIRSmall",
            "LeCaRDv2",
            "LegalBenchConsumerContractsQA",
            "LegalBenchCorporateLobbying",
            "LegalQuAD",
        ]
    ),
    description="A benchmark of retrieval tasks in the legal domain.",
    reference=None,
    citation=None,
)

MTEB_RETRIEVAL_MEDICAL = Benchmark(
    name="MTEB(Medical, v1)",
    aliases=["MTEB(Medical)"],
    display_name="Medical",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-hospital.svg",
    tasks=get_tasks(
        tasks=[
            "CUREv1",
            "NFCorpus",
            "TRECCOVID",
            "TRECCOVID-PL",
            "SciFact",
            "SciFact-PL",
            "MedicalQARetrieval",
            "PublicHealthQA",
            "MedrxivClusteringP2P.v2",
            "MedrxivClusteringS2S.v2",
            "CmedqaRetrieval",
            "CMedQAv2-reranking",
        ],
    ),
    description="A curated set of MTEB tasks designed to evaluate systems in the context of medical information retrieval.",
    reference=None,
    citation=None,
)

MTEB_MINERS_BITEXT_MINING = Benchmark(
    name="MINERSBitextMining",
    tasks=get_tasks(
        tasks=[
            "BUCC",
            "LinceMTBitextMining",
            "NollySentiBitextMining",
            "NusaXBitextMining",
            "NusaTranslationBitextMining",
            "PhincBitextMining",
            "Tatoeba",
        ]
    ),
    description="""Bitext Mining texts from the MINERS benchmark, a benchmark designed to evaluate the
    ability of multilingual LMs in semantic retrieval tasks,
    including bitext mining and classification via retrieval-augmented contexts.
    """,
    reference="https://arxiv.org/pdf/2406.07424",
    citation=r"""
@article{winata2024miners,
  author = {Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
  journal = {arXiv preprint arXiv:2406.07424},
  title = {MINERS: Multilingual Language Models as Semantic Retrievers},
  year = {2024},
}
""",
)

SEB = Benchmark(
    name="MTEB(Scandinavian, v1)",
    aliases=["MTEB(Scandinavian)", "SEB"],
    display_name="Scandinavian",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/dk.svg",
    language_view=["dan-Latn", "swe-Latn", "nno-Latn", "nob-Latn"],
    tasks=get_tasks(
        tasks=[
            # Bitext
            "BornholmBitextMining",
            "NorwegianCourtsBitextMining",
            # Classification
            "AngryTweetsClassification",
            "DanishPoliticalCommentsClassification",
            "DalajClassification",
            "DKHateClassification",
            "LccSentimentClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "NordicLangClassification",
            "NoRecClassification",
            "NorwegianParliamentClassification",
            "ScalaClassification",
            "SwedishSentimentClassification",
            "SweRecClassification",
            # Retrieval
            "DanFeverRetrieval",
            "NorQuadRetrieval",
            "SNLRetrieval",
            "SwednRetrieval",
            "SweFaqRetrieval",
            "TV2Nordretrieval",
            "TwitterHjerneRetrieval",
            # Clustering
            "SNLHierarchicalClusteringS2S",
            "SNLHierarchicalClusteringP2P",
            "SwednClusteringP2P",
            "SwednClusteringS2S",
            "VGHierarchicalClusteringS2S",
            "VGHierarchicalClusteringP2P",
        ],
        languages=["dan", "swe", "nno", "nob"],
    ),
    description="A curated selection of tasks coverering the Scandinavian languages; Danish, Swedish and Norwegian, including Bokmål and Nynorsk.",
    reference="https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/",
    citation=r"""
@inproceedings{enevoldsen2024scandinavian,
  author = {Enevoldsen, Kenneth and Kardos, M{\'a}rton and Muennighoff, Niklas and Nielbo, Kristoffer},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding},
  url = {https://nips.cc/virtual/2024/poster/97869},
  year = {2024},
}
""",
    contacts=["KennethEnevoldsen", "x-tabdeveloping", "Samoed"],
)

CoIR = Benchmark(
    name="CoIR",
    display_name="Code Information Retrieval",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            "AppsRetrieval",
            "CodeFeedbackMT",
            "CodeFeedbackST",
            "CodeSearchNetCCRetrieval",
            "CodeTransOceanContest",
            "CodeTransOceanDL",
            "CosQA",
            "COIRCodeSearchNetRetrieval",
            "StackOverflowQA",
            "SyntheticText2SQL",
        ]
    ),
    description="CoIR: A Comprehensive Benchmark for Code Information Retrieval Models",
    reference="https://github.com/CoIR-team/coir",
    citation=r"""
@misc{li2024coircomprehensivebenchmarkcode,
  archiveprefix = {arXiv},
  author = {Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
  eprint = {2407.02883},
  primaryclass = {cs.IR},
  title = {CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
  url = {https://arxiv.org/abs/2407.02883},
  year = {2024},
}
""",
)

RAR_b = Benchmark(
    name="RAR-b",
    display_name="Reasoning as retrieval",
    tasks=get_tasks(
        tasks=[
            "ARCChallenge",
            "AlphaNLI",
            "HellaSwag",
            "WinoGrande",
            "PIQA",
            "SIQA",
            "Quail",
            "SpartQA",
            "TempReasonL1",
            "TempReasonL2Pure",
            "TempReasonL2Fact",
            "TempReasonL2Context",
            "TempReasonL3Pure",
            "TempReasonL3Fact",
            "TempReasonL3Context",
            "RARbCode",
            "RARbMath",
        ]
    ),
    description="A benchmark to evaluate reasoning capabilities of retrievers.",
    reference="https://arxiv.org/abs/2404.06347",
    citation=r"""
@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Al Moubayed, Noura},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
    contacts=["gowitheflow-1998"],
)

MTEB_FRA = Benchmark(
    name="MTEB(fra, v1)",
    aliases=["MTEB(fra)"],
    display_name="French",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/fr.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["fra"],
            tasks=[
                # Classification
                "AmazonReviewsClassification",
                "MasakhaNEWSClassification",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "MTOPDomainClassification",
                "MTOPIntentClassification",
                # Clustering
                "AlloProfClusteringP2P",
                "AlloProfClusteringS2S",
                "HALClusteringS2S",
                "MasakhaNEWSClusteringP2P",
                "MasakhaNEWSClusteringS2S",
                "MLSUMClusteringP2P",
                "MLSUMClusteringS2S",
                # Pair Classification
                "PawsXPairClassification",
                # Reranking
                "AlloprofReranking",
                "SyntecReranking",
                # Retrieval
                "AlloprofRetrieval",
                "BSARDRetrieval",
                "MintakaRetrieval",
                "SyntecRetrieval",
                "XPQARetrieval",
                # STS
                "SICKFr",
                "STSBenchmarkMultilingualSTS",
                "SummEvalFr",
            ],
        )
        + (get_task("STS22", eval_splits=["test"], hf_subsets=["fr"]),)
    ),
    description="MTEB-French, a French expansion of the original benchmark with high-quality native French datasets.",
    reference="https://arxiv.org/abs/2405.20468",
    citation=r"""
@misc{ciancone2024mtebfrenchresourcesfrenchsentence,
  archiveprefix = {arXiv},
  author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
  eprint = {2405.20468},
  primaryclass = {cs.CL},
  title = {MTEB-French: Resources for French Sentence Embedding Evaluation and Analysis},
  url = {https://arxiv.org/abs/2405.20468},
  year = {2024},
}
""",
    contacts=["imenelydiaker"],
)

MTEB_DEU = Benchmark(
    name="MTEB(deu, v1)",
    aliases=["MTEB(deu)"],
    display_name="German",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/de.svg",
    tasks=get_tasks(
        languages=["deu"],
        exclusive_language_filter=True,
        tasks=[
            # Classification
            "AmazonCounterfactualClassification",
            "AmazonReviewsClassification",
            "MTOPDomainClassification",
            "MTOPIntentClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            # Clustering
            "BlurbsClusteringP2P",
            "BlurbsClusteringS2S",
            "TenKGnadClusteringP2P",
            "TenKGnadClusteringS2S",
            # Pair Classification
            "FalseFriendsGermanEnglish",
            "PawsXPairClassification",
            # Reranking
            "MIRACLReranking",
            # Retrieval
            "GermanQuAD-Retrieval",
            "GermanDPR",
            "XMarket",
            "GerDaLIR",
            # STS
            "GermanSTSBenchmark",
            "STS22",
        ],
    ),
    description="A benchmark for text-embedding performance in German.",
    reference="https://arxiv.org/html/2401.02709v1",
    citation=r"""
@misc{wehrli2024germantextembeddingclustering,
  archiveprefix = {arXiv},
  author = {Silvan Wehrli and Bert Arnrich and Christopher Irrgang},
  eprint = {2401.02709},
  primaryclass = {cs.CL},
  title = {German Text Embedding Clustering Benchmark},
  url = {https://arxiv.org/abs/2401.02709},
  year = {2024},
}
""",
    contacts=["slvnwhrl"],
)

MTEB_KOR = Benchmark(
    name="MTEB(kor, v1)",
    aliases=["MTEB(kor)"],
    display_name="Korean",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/kr.svg",
    tasks=get_tasks(
        languages=["kor"],
        tasks=[  # @KennethEnevoldsen: We could probably expand this to a more solid benchmark, but for now I have left it as is.
            # Classification
            "KLUE-TC",
            # Reranking
            "MIRACLReranking",
            # Retrieval
            "MIRACLRetrieval",
            "Ko-StrategyQA",
            # STS
            "KLUE-STS",
            "KorSTS",
        ],
    ),
    description="A benchmark and leaderboard for evaluation of text embedding in Korean.",
    reference=None,
    citation=None,
)

MTEB_POL = Benchmark(
    name="MTEB(pol, v1)",
    aliases=["MTEB(pol)"],
    display_name="Polish",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/pl.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["pol"],
            tasks=[
                # Classification
                "AllegroReviews",
                "CBD",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "PolEmo2.0-IN",
                "PolEmo2.0-OUT",
                "PAC",
                # Clustering
                "EightTagsClustering",
                "PlscClusteringS2S",
                "PlscClusteringP2P",
                # Pair Classification
                "CDSC-E",
                "PpcPC",
                "PSC",
                "SICK-E-PL",
                # STS
                "CDSC-R",
                "SICK-R-PL",
            ],
        )
        + (get_task("STS22", eval_splits=["test"], hf_subsets=["pl"]),),
    ),
    description="""Polish Massive Text Embedding Benchmark (PL-MTEB), a comprehensive benchmark for text embeddings in Polish. The PL-MTEB consists of 28 diverse NLP
tasks from 5 task types. With tasks adapted based on previously used datasets by the Polish
NLP community. In addition, a new PLSC (Polish Library of Science Corpus) dataset was created
consisting of titles and abstracts of scientific publications in Polish, which was used as the basis for
two novel clustering tasks.""",  # Rephrased from the abstract
    reference="https://arxiv.org/abs/2405.10138",
    citation=r"""
@article{poswiata2024plmteb,
  author = {Rafał Poświata and Sławomir Dadas and Michał Perełkiewicz},
  journal = {arXiv preprint arXiv:2405.10138},
  title = {PL-MTEB: Polish Massive Text Embedding Benchmark},
  year = {2024},
}
""",
    contacts=["rafalposwiata"],
)

MTEB_code = Benchmark(
    name="MTEB(Code, v1)",
    aliases=["MTEB(code)"],
    display_name="Code",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            # Retrieval
            "AppsRetrieval",
            "CodeEditSearchRetrieval",
            "CodeFeedbackMT",
            "CodeFeedbackST",
            "CodeSearchNetCCRetrieval",
            "CodeSearchNetRetrieval",
            "CodeTransOceanContest",
            "CodeTransOceanDL",
            "CosQA",
            "COIRCodeSearchNetRetrieval",
            "StackOverflowQA",
            "SyntheticText2SQL",
        ],
    ),
    description="A massive code embedding benchmark covering retrieval tasks in a miriad of popular programming languages.",
    reference=None,
    citation=MMTEB_CITATION,
)

mteb_multilingual_tasks = get_tasks(
    tasks=[
        "BornholmBitextMining",
        "BibleNLPBitextMining",
        "BUCC.v2",
        "DiaBlaBitextMining",
        "FloresBitextMining",
        "IN22GenBitextMining",
        "IndicGenBenchFloresBitextMining",
        "NollySentiBitextMining",
        "NorwegianCourtsBitextMining",
        "NTREXBitextMining",
        "NusaTranslationBitextMining",
        "NusaXBitextMining",
        "Tatoeba",
        "BulgarianStoreReviewSentimentClassfication",
        "CzechProductReviewSentimentClassification",
        "GreekLegalCodeClassification",
        "DBpediaClassification",
        "FinancialPhrasebankClassification",
        "PoemSentimentClassification",
        "ToxicConversationsClassification",
        "TweetTopicSingleClassification",
        "EstonianValenceClassification",
        "FilipinoShopeeReviewsClassification",
        "GujaratiNewsClassification",
        "SentimentAnalysisHindi",
        "IndonesianIdClickbaitClassification",
        "ItaCaseholdClassification",
        "KorSarcasmClassification",
        "KurdishSentimentClassification",
        "MacedonianTweetSentimentClassification",
        "AfriSentiClassification",
        "AmazonCounterfactualClassification",
        "CataloniaTweetClassification",
        "CyrillicTurkicLangClassification",
        "IndicLangClassification",
        "MasakhaNEWSClassification",
        "MassiveIntentClassification",
        "MultiHateClassification",
        "NordicLangClassification",
        "NusaParagraphEmotionClassification",
        "NusaX-senti",
        "ScalaClassification",
        "SwissJudgementClassification",
        "NepaliNewsClassification",
        "OdiaNewsClassification",
        "PunjabiNewsClassification",
        "PolEmo2.0-OUT",
        "PAC",
        "SinhalaNewsClassification",
        "CSFDSKMovieReviewSentimentClassification",
        "SiswatiNewsClassification",
        "SlovakMovieReviewSentimentClassification",
        "SwahiliNewsClassification",
        "DalajClassification",
        "TswanaNewsClassification",
        "IsiZuluNewsClassification",
        "WikiCitiesClustering",
        "MasakhaNEWSClusteringS2S",
        "RomaniBibleClustering",
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        "BigPatentClustering.v2",
        "BiorxivClusteringP2P.v2",
        "MedrxivClusteringP2P.v2",
        "StackExchangeClustering.v2",
        "AlloProfClusteringS2S.v2",
        "HALClusteringS2S.v2",
        "SIB200ClusteringS2S",
        "WikiClusteringP2P.v2",
        "PlscClusteringP2P.v2",
        "SwednClusteringP2P",
        "CLSClusteringP2P.v2",
        "StackOverflowQA",
        "TwitterHjerneRetrieval",
        "AILAStatutes",
        "ArguAna",
        "HagridRetrieval",
        "LegalBenchCorporateLobbying",
        "LEMBPasskeyRetrieval",
        "SCIDOCS",
        "SpartQA",
        "TempReasonL1",
        "TRECCOVID",
        "WinoGrande",
        "BelebeleRetrieval",
        "MLQARetrieval",
        "StatcanDialogueDatasetRetrieval",
        "WikipediaRetrievalMultilingual",
        "CovidRetrieval",
        "Core17InstructionRetrieval",
        "News21InstructionRetrieval",
        "Robust04InstructionRetrieval",
        "KorHateSpeechMLClassification",
        "MalteseNewsClassification",
        "MultiEURLEXMultilabelClassification",
        "BrazilianToxicTweetsClassification",
        "CEDRClassification",
        "CTKFactsNLI",
        "SprintDuplicateQuestions",
        "TwitterURLCorpus",
        "ArmenianParaphrasePC",
        "indonli",
        "OpusparcusPC",
        "PawsXPairClassification",
        "RTE3",
        "XNLI",
        "PpcPC",
        "TERRa",
        "WebLINXCandidatesReranking",
        "AlloprofReranking",
        "VoyageMMarcoReranking",
        "WikipediaRerankingMultilingual",
        "RuBQReranking",
        "T2Reranking",
        "GermanSTSBenchmark",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STSBenchmark",
        "FaroeseSTS",
        "FinParaSTS",
        "JSICK",
        "IndicCrosslingualSTS",
        "SemRel24STS",
        "STS17",
        "STS22.v2",
        "STSES",
        "STSB",
        "MIRACLRetrievalHardNegatives",
    ],
)

MTEB_multilingual_v1 = Benchmark(
    name="MTEB(Multilingual, v1)",
    display_name="Multilingual",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-globe.svg",
    tasks=MTEBTasks(
        mteb_multilingual_tasks + get_tasks(tasks=["SNLHierarchicalClusteringP2P"])
    ),
    description="A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages. This benhcmark has been replaced by MTEB(Multilingual, v2) as one of the datasets (SNLHierarchicalClustering) included in v1 was removed from the Hugging Face Hub.",
    reference="https://arxiv.org/abs/2502.13595",
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)


MTEB_multilingual_v2 = Benchmark(
    name="MTEB(Multilingual, v2)",
    aliases=["MTEB(Multilingual)", "MMTEB"],
    display_name="Multilingual",
    language_view=[
        "eng-Latn",  # English
        "zho-Hans",  # Chinese (Simplified)
        "hin-Deva",  # Hindi
        "spa-Latn",  # Spanish
        "fra-Latn",  # French
        "ara-Arab",  # Arabic
        "ben-Beng",  # Bengali
        "rus-Cyrl",  # Russian
        "por-Latn",  # Portuguese
        "urd-Arab",  # Urdu
        "ind-Latn",  # Indonesian
        "deu-Latn",  # German
        "jpn-Jpan",  # Japanese
        "swa-Latn",  # Swahili
        "mar-Deva",  # Marathi
        "tel-Telu",  # Telugu
        "tur-Latn",  # Turkish
        "tam-Taml",  # Tamil
        "vie-Latn",  # Vietnamese
        "kor-Hang",  # Korean
    ],
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-globe.svg",
    tasks=mteb_multilingual_tasks,
    description="A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages. ",
    reference="https://arxiv.org/abs/2502.13595",
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)

MTEB_JPN = Benchmark(
    name="MTEB(jpn, v1)",
    aliases=["MTEB(jpn)"],
    display_name="Japanese Legacy",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/jp.svg",
    tasks=get_tasks(
        languages=["jpn"],
        tasks=[
            # clustering
            "LivedoorNewsClustering.v2",
            "MewsC16JaClustering",
            # classification
            "AmazonReviewsClassification",
            "AmazonCounterfactualClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            # STS
            "JSTS",
            "JSICK",
            # pair classification
            "PawsXPairClassification",
            # retrieval
            "JaqketRetrieval",
            "MrTidyRetrieval",
            "JaGovFaqsRetrieval",
            "NLPJournalTitleAbsRetrieval",
            "NLPJournalAbsIntroRetrieval",
            "NLPJournalTitleIntroRetrieval",
            # reranking
            "ESCIReranking",
        ],
    ),
    description="JMTEB is a benchmark for evaluating Japanese text embedding models.",
    reference="https://github.com/sbintuitions/JMTEB",
    citation=None,
)


indic_languages = [
    "asm",
    "awa",
    "ben",
    "bgc",
    "bho",
    "doi",
    "gbm",
    "gom",
    "guj",
    "hin",
    "hne",
    "kan",
    "kas",
    "mai",
    "mal",
    "mar",
    "mni",
    "mup",
    "mwr",
    "nep",
    "npi",
    "ori",
    "ory",
    "pan",
    "raj",
    "san",
    "snd",
    "tam",
    "tel",
    "urd",
]

MTEB_INDIC = Benchmark(
    name="MTEB(Indic, v1)",
    aliases=["MTEB(Indic)"],
    display_name="Indic",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/in.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                # Bitext
                "IN22ConvBitextMining",
                "IN22GenBitextMining",
                # clustering
                "SIB200ClusteringS2S",
                # classification
                "BengaliSentimentAnalysis",
                "GujaratiNewsClassification",
                "HindiDiscourseClassification",
                "SentimentAnalysisHindi",
                "MalayalamNewsClassification",
                "MTOPIntentClassification",
                "MultiHateClassification",
                "TweetSentimentClassification",
                "NepaliNewsClassification",
                "PunjabiNewsClassification",
                "SanskritShlokasClassification",
                "UrduRomanSentimentClassification",
                # pair classification
                "XNLI",
                # retrieval
                "BelebeleRetrieval",
                "XQuADRetrieval",
                # reranking
                "WikipediaRerankingMultilingual",
            ],
            languages=indic_languages,
            exclusive_language_filter=True,
        )
        +
        # STS
        (get_task("IndicCrosslingualSTS"),)
    ),
    description="A regional geopolitical text embedding benchmark targeting embedding performance on Indic languages.",
    reference=None,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)


eu_languages = [
    # official EU languages (56) - we could include the whole economic area e.g. Norway - additionally we could include minority languages (probably a good idea?)
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
    # Schengen Area
    "nno",
    "nob",
    "isl",
    "ron",
    "eus",  # Basque - recognized minority language
    "ron",  # Romanian - recognized minority language
    "rom",  # Romani - recognized minority language
]

MTEB_EU = Benchmark(
    name="MTEB(Europe, v1)",
    aliases=["MTEB(Europe)"],
    display_name="European",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/eu.svg",
    tasks=get_tasks(
        tasks=[
            "BornholmBitextMining",
            "BibleNLPBitextMining",
            "BUCC.v2",
            "DiaBlaBitextMining",
            "FloresBitextMining",
            "NorwegianCourtsBitextMining",
            "NTREXBitextMining",
            "BulgarianStoreReviewSentimentClassfication",
            "CzechProductReviewSentimentClassification",
            "GreekLegalCodeClassification",
            "DBpediaClassification",
            "FinancialPhrasebankClassification",
            "PoemSentimentClassification",
            "ToxicChatClassification",
            "ToxicConversationsClassification",
            "EstonianValenceClassification",
            "ItaCaseholdClassification",
            "AmazonCounterfactualClassification",
            "MassiveScenarioClassification",
            "MultiHateClassification",
            "ScalaClassification",
            "SwissJudgementClassification",
            "TweetSentimentClassification",
            "CBD",
            "PolEmo2.0-OUT",
            "CSFDSKMovieReviewSentimentClassification",
            "DalajClassification",
            "WikiCitiesClustering",
            "RomaniBibleClustering",
            "BigPatentClustering.v2",
            "BiorxivClusteringP2P.v2",
            "AlloProfClusteringS2S.v2",
            "HALClusteringS2S.v2",
            "SIB200ClusteringS2S",
            "WikiClusteringP2P.v2",
            "StackOverflowQA",
            "TwitterHjerneRetrieval",
            "LegalQuAD",
            "ArguAna",
            "HagridRetrieval",
            "LegalBenchCorporateLobbying",
            "LEMBPasskeyRetrieval",
            "SCIDOCS",
            "SpartQA",
            "TempReasonL1",
            "WinoGrande",
            "AlloprofRetrieval",
            "BelebeleRetrieval",
            "StatcanDialogueDatasetRetrieval",
            "WikipediaRetrievalMultilingual",
            "Core17InstructionRetrieval",
            "News21InstructionRetrieval",
            "Robust04InstructionRetrieval",
            "MalteseNewsClassification",
            "MultiEURLEXMultilabelClassification",
            "CTKFactsNLI",
            "SprintDuplicateQuestions",
            "OpusparcusPC",
            "RTE3",
            "XNLI",
            "PSC",
            "WebLINXCandidatesReranking",
            "AlloprofReranking",
            "WikipediaRerankingMultilingual",
            "SICK-R",
            "STS12",
            "STS14",
            "STS15",
            "STSBenchmark",
            "FinParaSTS",
            "STS17",
            "SICK-R-PL",
            "STSES",
        ],
        languages=eu_languages,
        exclusive_language_filter=True,
    ),
    description="A regional geopolitical text embedding benchmark targeting embedding performance on European languages.",
    reference=None,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)

LONG_EMBED = Benchmark(
    name="LongEmbed",
    display_name="Long-context Retrieval",
    tasks=get_tasks(
        tasks=[
            "LEMBNarrativeQARetrieval",
            "LEMBNeedleRetrieval",
            "LEMBPasskeyRetrieval",
            "LEMBQMSumRetrieval",
            "LEMBSummScreenFDRetrieval",
            "LEMBWikimQARetrieval",
        ],
    ),
    description="""LongEmbed is a benchmark oriented at exploring models' performance on long-context retrieval.
    The benchmark comprises two synthetic tasks and four carefully chosen real-world tasks,
    featuring documents of varying length and dispersed target information.
    """,  # Pieced together from paper abstract.
    reference="https://arxiv.org/abs/2404.12096v2",
    citation=r"""
@article{zhu2024longembed,
  author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
  journal = {arXiv preprint arXiv:2404.12096},
  title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
  year = {2024},
}
""",
)

BRIGHT = Benchmark(
    name="BRIGHT",
    display_name="Reasoning Retrieval",
    tasks=get_tasks(tasks=["BrightRetrieval"], eval_splits=["standard"]),
    description="""BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
    BRIGHT is the first text retrieval
    benchmark that requires intensive reasoning to retrieve relevant documents with
    a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
    economics, psychology, mathematics, and coding. These queries are drawn from
    naturally occurring and carefully curated human data.
    """,
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
)

BRIGHT_LONG = Benchmark(
    name="BRIGHT (long)",
    aliases=["BRIGHT(long)"],
    tasks=MTEBTasks(
        (
            get_task(
                "BrightLongRetrieval",
            ),
        )
    ),
    description="""BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
BRIGHT is the first text retrieval
benchmark that requires intensive reasoning to retrieve relevant documents with
a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
economics, psychology, mathematics, and coding. These queries are drawn from
naturally occurring and carefully curated human data.

This is the long version of the benchmark, which only filter longer documents.
    """,
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
)

CODE_RAG = Benchmark(
    name="CodeRAG",
    tasks=get_tasks(
        tasks=[
            "CodeRAGLibraryDocumentationSolutions",
            "CodeRAGOnlineTutorials",
            "CodeRAGProgrammingSolutions",
            "CodeRAGStackoverflowPosts",
        ],
    ),
    description="A benchmark for evaluating code retrieval augmented generation, testing models' ability to retrieve relevant programming solutions, tutorials and documentation.",
    reference="https://arxiv.org/abs/2406.14497",
    citation=r"""
@misc{wang2024coderagbenchretrievalaugmentcode,
  archiveprefix = {arXiv},
  author = {Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
  eprint = {2406.14497},
  primaryclass = {cs.SE},
  title = {CodeRAG-Bench: Can Retrieval Augment Code Generation?},
  url = {https://arxiv.org/abs/2406.14497},
  year = {2024},
}
""",
)

BEIR = Benchmark(
    name="BEIR",
    display_name="BEIR",
    icon="https://github.com/lipis/flag-icons/raw/refs/heads/main/flags/4x3/us.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "TRECCOVID",
                "NFCorpus",
                "NQ",
                "HotpotQA",
                "FiQA2018",
                "ArguAna",
                "Touche2020",
                "CQADupstackRetrieval",
                "QuoraRetrieval",
                "DBPedia",
                "SCIDOCS",
                "FEVER",
                "ClimateFEVER",
                "SciFact",
            ],
        )
        + get_tasks(tasks=["MSMARCO"], languages=["eng"], eval_splits=["dev"])
    ),
    description="BEIR is a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your NLP-based retrieval models within the benchmark.",
    reference="https://arxiv.org/abs/2104.08663",
    citation=r"""
@article{thakur2021beir,
  author = {Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
  journal = {arXiv preprint arXiv:2104.08663},
  title = {Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models},
  year = {2021},
}
""",
)

NANOBEIR = Benchmark(
    name="NanoBEIR",
    tasks=get_tasks(
        tasks=[
            "NanoArguAnaRetrieval",
            "NanoClimateFeverRetrieval",
            "NanoDBPediaRetrieval",
            "NanoFEVERRetrieval",
            "NanoFiQA2018Retrieval",
            "NanoHotpotQARetrieval",
            "NanoMSMARCORetrieval",
            "NanoNFCorpusRetrieval",
            "NanoNQRetrieval",
            "NanoQuoraRetrieval",
            "NanoSCIDOCSRetrieval",
            "NanoSciFactRetrieval",
            "NanoTouche2020Retrieval",
        ],
    ),
    description="A benchmark to evaluate with subsets of BEIR datasets to use less computational power",
    reference="https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6",
    citation=None,
)

C_MTEB = Benchmark(
    name="MTEB(cmn, v1)",
    aliases=["MTEB(Chinese)", "CMTEB"],
    display_name="Chinese",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/cn.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "T2Retrieval",
                "MMarcoRetrieval",
                "DuRetrieval",
                "CovidRetrieval",
                "CmedqaRetrieval",
                "EcomRetrieval",
                "MedicalRetrieval",
                "VideoRetrieval",
                "T2Reranking",
                "MMarcoReranking",
                "CMedQAv1-reranking",
                "CMedQAv2-reranking",
                "Ocnli",
                "Cmnli",
                "CLSClusteringS2S",
                "CLSClusteringP2P",
                "ThuNewsClusteringS2S",
                "ThuNewsClusteringP2P",
                "LCQMC",
                "PAWSX",
                "AFQMC",
                "QBQTC",
                "TNews",
                "IFlyTek",
                "Waimai",
                "OnlineShopping",
                "JDReview",
            ],
        )
        + get_tasks(
            tasks=[
                "MultilingualSentiment",
                "ATEC",
                "BQ",
                "STSB",
            ],
            eval_splits=["test"],
        )
        + get_tasks(
            tasks=[
                "MultilingualSentiment",
            ],
            eval_splits=["validation"],
        )
    ),
    description="The Chinese Massive Text Embedding Benchmark (C-MTEB) is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets.",
    reference="https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB",
    citation=r"""
@misc{c-pack,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  year = {2023},
}
""",
)

FA_MTEB = Benchmark(
    name="MTEB(fas, v1)",
    aliases=["FaMTEB(fas, beta)"],
    display_name="Farsi Legacy",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/ir.svg",
    tasks=get_tasks(
        languages=["fas"],
        tasks=[
            # Classification
            "PersianFoodSentimentClassification",
            "SynPerChatbotConvSAClassification",
            "SynPerChatbotConvSAToneChatbotClassification",
            "SynPerChatbotConvSAToneUserClassification",
            "SynPerChatbotSatisfactionLevelClassification",
            "SynPerChatbotRAGToneChatbotClassification",
            "SynPerChatbotRAGToneUserClassification",
            "SynPerChatbotToneChatbotClassification",
            "SynPerChatbotToneUserClassification",
            "SynPerTextToneClassification",
            "SIDClassification",
            "DeepSentiPers",
            "PersianTextEmotion",
            "SentimentDKSF",
            "NLPTwitterAnalysisClassification",
            "DigikalamagClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            # Clustering
            "BeytooteClustering",
            "DigikalamagClustering",
            "HamshahriClustring",
            "NLPTwitterAnalysisClustering",
            "SIDClustring",
            # PairClassification
            "FarsTail",
            "CExaPPC",
            "SynPerChatbotRAGFAQPC",
            "FarsiParaphraseDetection",
            "SynPerTextKeywordsPC",
            "SynPerQAPC",
            "ParsinluEntail",
            "ParsinluQueryParaphPC",
            # Reranking
            "MIRACLReranking",
            "WikipediaRerankingMultilingual",
            # Retrieval
            "SynPerQARetrieval",
            "SynPerChatbotTopicsRetrieval",
            "SynPerChatbotRAGTopicsRetrieval",
            "SynPerChatbotRAGFAQRetrieval",
            "PersianWebDocumentRetrieval",
            "WikipediaRetrievalMultilingual",
            "MIRACLRetrieval",
            "ClimateFEVER-Fa",
            "DBPedia-Fa",
            "HotpotQA-Fa",
            "MSMARCO-Fa",
            "NQ-Fa",
            "ArguAna-Fa",
            "CQADupstackRetrieval-Fa",
            "FiQA2018-Fa",
            "NFCorpus-Fa",
            "QuoraRetrieval-Fa",
            "SCIDOCS-Fa",
            "SciFact-Fa",
            "TRECCOVID-Fa",
            "Touche2020-Fa",
            # STS
            "Farsick",
            "SynPerSTS",
            "Query2Query",
            # SummaryRetrieval
            "SAMSumFa",
            "SynPerChatbotSumSRetrieval",
            "SynPerChatbotRAGSumSRetrieval",
        ],
    ),
    description="The Persian Massive Text Embedding Benchmark (FaMTEB) is a comprehensive benchmark for Persian text embeddings covering 7 tasks and 60+ datasets.",
    reference="https://arxiv.org/abs/2502.11571",
    citation=r"""
@article{zinvandi2025famteb,
  author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
  journal = {arXiv preprint arXiv:2502.11571},
  title = {Famteb: Massive text embedding benchmark in persian language},
  year = {2025},
}
""",
    contacts=["mehran-sarmadi", "ERfun", "morteza20"],
)

FA_MTEB_2 = Benchmark(
    name="MTEB(fas, v2)",
    display_name="Farsi",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/ir.svg",
    tasks=get_tasks(
        languages=["fas"],
        tasks=[
            # Classification
            "PersianFoodSentimentClassification",
            "SynPerChatbotConvSAClassification",
            "SynPerChatbotConvSAToneChatbotClassification",
            "SynPerChatbotConvSAToneUserClassification",
            "SynPerChatbotSatisfactionLevelClassification",
            "SynPerTextToneClassification.v3",
            "SIDClassification.v2",
            "DeepSentiPers.v2",
            "PersianTextEmotion.v2",
            "NLPTwitterAnalysisClassification.v2",
            "DigikalamagClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "StyleClassification",
            "PerShopDomainClassification",
            "PerShopIntentClassification",
            # Clustering
            "BeytooteClustering",
            "DigikalamagClustering",
            "HamshahriClustring",
            "NLPTwitterAnalysisClustering",
            "SIDClustring",
            # PairClassification
            "FarsTail",
            "SynPerChatbotRAGFAQPC",
            "FarsiParaphraseDetection",
            "SynPerTextKeywordsPC",
            "SynPerQAPC",
            "ParsinluEntail",
            "ParsinluQueryParaphPC",
            # Reranking
            "MIRACLReranking",
            "WikipediaRerankingMultilingual",
            # Retrieval
            "SynPerQARetrieval",
            "SynPerChatbotRAGFAQRetrieval",
            "PersianWebDocumentRetrieval",
            "WikipediaRetrievalMultilingual",
            "MIRACLRetrievalHardNegatives",
            "HotpotQA-FaHardNegatives",
            "MSMARCO-FaHardNegatives",
            "NQ-FaHardNegatives",
            "ArguAna-Fa.v2",
            "FiQA2018-Fa.v2",
            "QuoraRetrieval-Fa.v2",
            "SCIDOCS-Fa.v2",
            "SciFact-Fa.v2",
            "TRECCOVID-Fa.v2",
            "FEVER-FaHardNegatives",
            "NeuCLIR2023RetrievalHardNegatives",
            "WebFAQRetrieval",
            # STS
            "Farsick",
            "SynPerSTS",
            # SummaryRetrieval
            "SAMSumFa",
            "SynPerChatbotSumSRetrieval",
            "SynPerChatbotRAGSumSRetrieval",
        ],
    ),
    description="The Persian Massive Text Embedding Benchmark (FaMTEB) is a comprehensive benchmark for Persian text embeddings covering 7 tasks and 50+ datasets. In version 2, we have optimized large datasets to make them more manageable and accessible, removed low-quality datasets, and added higher-quality data to improve the overall benchmark. For more details on the improvements, see the main PR comment: [main PR](https://github.com/embeddings-benchmark/mteb/pull/3157).",
    reference="https://arxiv.org/abs/2502.11571",
    citation=r"""
@article{zinvandi2025famteb,
  author = {Zinvandi, Erfan and Alikhani, Morteza and Sarmadi, Mehran and Pourbahman, Zahra and Arvin, Sepehr and Kazemi, Reza and Amini, Arash},
  journal = {arXiv preprint arXiv:2502.11571},
  title = {Famteb: Massive text embedding benchmark in persian language},
  year = {2025},
}
""",
    contacts=["mehran-sarmadi", "ERfun", "morteza20"],
)

CHEMTEB = Benchmark(
    name="ChemTEB",
    display_name="Chemical",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-purge.svg",
    tasks=get_tasks(
        tasks=[
            "PubChemSMILESBitextMining",
            "SDSEyeProtectionClassification",
            "SDSGlovesClassification",
            "WikipediaBioMetChemClassification",
            "WikipediaGreenhouseEnantiopureClassification",
            "WikipediaSolidStateColloidalClassification",
            "WikipediaOrganicInorganicClassification",
            "WikipediaCryobiologySeparationClassification",
            "WikipediaChemistryTopicsClassification",
            "WikipediaTheoreticalAppliedClassification",
            "WikipediaChemFieldsClassification",
            "WikipediaLuminescenceClassification",
            "WikipediaIsotopesFissionClassification",
            "WikipediaSaltsSemiconductorsClassification",
            "WikipediaBiolumNeurochemClassification",
            "WikipediaCrystallographyAnalyticalClassification",
            "WikipediaCompChemSpectroscopyClassification",
            "WikipediaChemEngSpecialtiesClassification",
            "WikipediaChemistryTopicsClustering",
            "WikipediaSpecialtiesInChemistryClustering",
            "PubChemAISentenceParaphrasePC",
            "PubChemSMILESPC",
            "PubChemSynonymPC",
            "PubChemWikiParagraphsPC",
            "PubChemWikiPairClassification",
            "ChemNQRetrieval",
            "ChemHotpotQARetrieval",
        ],
    ),
    description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
    reference="https://arxiv.org/abs/2412.00532",
    citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \\& Efficiency on a Specific Domain},
  year = {2024},
}
""",
)

BEIR_NL = Benchmark(
    name="BEIR-NL",
    display_name="BEIR-NL",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/nl.svg",
    tasks=get_tasks(
        tasks=[
            "ArguAna-NL",
            "CQADupstack-NL",
            "FEVER-NL",
            "NQ-NL",
            "Touche2020-NL",
            "FiQA2018-NL",
            "Quora-NL",
            "HotpotQA-NL",
            "SCIDOCS-NL",
            "ClimateFEVER-NL",
            "mMARCO-NL",
            "SciFact-NL",
            "DBPedia-NL",
            "NFCorpus-NL",
            "TRECCOVID-NL",
        ],
    ),
    description="BEIR-NL is a Dutch adaptation of the publicly available BEIR benchmark, created through automated "
    "translation.",
    reference="https://arxiv.org/abs/2412.08329",
    contacts=["nikolay-banar"],
    citation=r"""
@misc{banar2024beirnlzeroshotinformationretrieval,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
  eprint = {2412.08329},
  primaryclass = {cs.CL},
  title = {BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
  url = {https://arxiv.org/abs/2412.08329},
  year = {2024},
}
""",
)

MTEB_NL = Benchmark(
    name="MTEB(nld, v1)",
    display_name="Dutch",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/nl.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["nld"],
            exclusive_language_filter=True,
            tasks=[
                # Classification
                "DutchBookReviewSentimentClassification.v2",
                "MassiveIntentClassification",
                "MassiveScenarioClassification",
                "SIB200Classification",
                "MultiHateClassification",
                "VaccinChatNLClassification",
                "DutchColaClassification",
                "DutchGovernmentBiasClassification",
                "DutchSarcasticHeadlinesClassification",
                "DutchNewsArticlesClassification",
                "OpenTenderClassification",
                "IconclassClassification",
                # # PairClassification
                "SICKNLPairClassification",
                "XLWICNLPairClassification",
                # # MultiLabelClassification
                "CovidDisinformationNLMultiLabelClassification",
                "MultiEURLEXMultilabelClassification",
                "VABBMultiLabelClassification",
                # # Clustering
                "DutchNewsArticlesClusteringS2S",
                "DutchNewsArticlesClusteringP2P",
                "SIB200ClusteringS2S",
                "VABBClusteringS2S",
                "VABBClusteringP2P",
                "OpenTenderClusteringS2S",
                "OpenTenderClusteringP2P",
                "IconclassClusteringS2S",
                # # Reranking
                "WikipediaRerankingMultilingual",
                # # Retrieval
                "ArguAna-NL.v2",
                "SCIDOCS-NL.v2",
                "SciFact-NL.v2",
                "NFCorpus-NL.v2",
                "BelebeleRetrieval",
                "WebFAQRetrieval",
                "DutchNewsArticlesRetrieval",
                "bBSARDNLRetrieval",
                "LegalQANLRetrieval",
                "OpenTenderRetrieval",
                "VABBRetrieval",
                "WikipediaRetrievalMultilingual",
                # # STS
                "SICK-NL-STS",
                "STSBenchmarkMultilingualSTS",
            ],
        )
    ),
    description="MTEB-NL",
    reference="https://arxiv.org/abs/2509.12340",
    contacts=["nikolay-banar"],
    citation=r"""
@misc{banar2025mtebnle5nlembeddingbenchmark,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
  eprint = {22509.12340},
  primaryclass = {cs.CL},
  title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
  url = {https://arxiv.org/abs/2509.12340},
  year = {2025},
}
""",
)

MIEB_common_tasks = [
    # Image Classification
    "Birdsnap",  # fine
    "Caltech101",  # fine
    "CIFAR10",  # coarse
    "CIFAR100",  # fine
    "Country211",  # fine
    "DTD",  # coarse
    "EuroSAT",  # coarse
    "FER2013",  # coarse
    "FGVCAircraft",  # fine
    "Food101Classification",  # fine
    "GTSRB",  # coarse
    "Imagenet1k",  # fine
    "MNIST",  # coarse
    "OxfordFlowersClassification",  # fine
    "OxfordPets",  # fine
    "PatchCamelyon",  # coarse
    "RESISC45",  # fine
    "StanfordCars",  # fine
    "STL10",  # coarse
    "SUN397",  # fine
    "UCF101",  # fine
    # ImageMultiLabelClassification
    "VOC2007",  # coarse
    # Clustering
    "CIFAR10Clustering",
    "CIFAR100Clustering",
    "ImageNetDog15Clustering",
    "ImageNet10Clustering",
    "TinyImageNetClustering",
    # ZeroShotClassification
    "BirdsnapZeroShot",
    "Caltech101ZeroShot",
    "CIFAR10ZeroShot",
    "CIFAR100ZeroShot",
    "CLEVRZeroShot",
    "CLEVRCountZeroShot",
    "Country211ZeroShot",
    "DTDZeroShot",
    "EuroSATZeroShot",
    "FER2013ZeroShot",
    "FGVCAircraftZeroShot",
    "Food101ZeroShot",
    "GTSRBZeroShot",
    "Imagenet1kZeroShot",
    "MNISTZeroShot",
    "OxfordPetsZeroShot",
    "PatchCamelyonZeroShot",
    "RenderedSST2",
    "RESISC45ZeroShot",
    "StanfordCarsZeroShot",
    "STL10ZeroShot",
    "SUN397ZeroShot",
    "UCF101ZeroShot",
    # Any2AnyMultipleChoice
    "BLINKIT2IMultiChoice",
    "BLINKIT2TMultiChoice",
    "CVBenchCount",
    "CVBenchRelation",
    "CVBenchDepth",
    "CVBenchDistance",
    # Compositionality
    "AROCocoOrder",
    "AROFlickrOrder",
    "AROVisualAttribution",
    "AROVisualRelation",
    "SugarCrepe",
    "Winoground",
    "ImageCoDe",
    # VisualSTS
    "STS12VisualSTS",
    "STS13VisualSTS",
    "STS14VisualSTS",
    "STS15VisualSTS",
    "STS16VisualSTS",
    # Any2AnyRetrieval
    "BLINKIT2IRetrieval",
    "BLINKIT2TRetrieval",
    "CIRRIT2IRetrieval",
    "CUB200I2IRetrieval",
    "EDIST2ITRetrieval",
    "Fashion200kI2TRetrieval",
    "Fashion200kT2IRetrieval",
    "FashionIQIT2IRetrieval",
    "Flickr30kI2TRetrieval",
    "Flickr30kT2IRetrieval",
    "FORBI2IRetrieval",
    "GLDv2I2IRetrieval",
    "GLDv2I2TRetrieval",
    "HatefulMemesI2TRetrieval",
    "HatefulMemesT2IRetrieval",
    "ImageCoDeT2IRetrieval",
    "InfoSeekIT2ITRetrieval",
    "InfoSeekIT2TRetrieval",
    "MemotionI2TRetrieval",
    "MemotionT2IRetrieval",
    "METI2IRetrieval",
    "MSCOCOI2TRetrieval",
    "MSCOCOT2IRetrieval",
    "NIGHTSI2IRetrieval",
    "OVENIT2ITRetrieval",
    "OVENIT2TRetrieval",
    "ROxfordEasyI2IRetrieval",
    "ROxfordMediumI2IRetrieval",
    "ROxfordHardI2IRetrieval",
    "RP2kI2IRetrieval",
    "RParisEasyI2IRetrieval",
    "RParisMediumI2IRetrieval",
    "RParisHardI2IRetrieval",
    "SciMMIRI2TRetrieval",
    "SciMMIRT2IRetrieval",
    "SketchyI2IRetrieval",
    "SOPI2IRetrieval",
    "StanfordCarsI2IRetrieval",
    "TUBerlinT2IRetrieval",
    "VidoreArxivQARetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreSyntheticDocQAEnergyRetrieval",
    "VidoreSyntheticDocQAGovernmentReportsRetrieval",
    "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
    "VisualNewsI2TRetrieval",
    "VisualNewsT2IRetrieval",
    "VizWizIT2TRetrieval",
    "VQA2IT2TRetrieval",
    "WebQAT2ITRetrieval",
    "WebQAT2TRetrieval",
]

MIEB_ENG = MIEBBenchmark(
    name="MIEB(eng)",
    display_name="Image-Text, English",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-picture.svg",
    tasks=get_tasks(
        tasks=MIEB_common_tasks
        + [
            "VisualSTS17Eng",
            "VisualSTS-b-Eng",
        ],
    ),
    description="""MIEB(eng) is a comprehensive image embeddings benchmark, spanning 8 task types, covering 125 tasks.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document understanding, visual STS, and CV-centric tasks.""",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  doi = {10.48550/ARXIV.2504.10471},
  journal = {arXiv preprint arXiv:2504.10471},
  publisher = {arXiv},
  title = {MIEB: Massive Image Embedding Benchmark},
  url = {https://arxiv.org/abs/2504.10471},
  year = {2025},
}
""",
)

MIEB_MULTILINGUAL = MIEBBenchmark(
    name="MIEB(Multilingual)",
    display_name="Image-Text, Multilingual",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-pictures.svg",
    tasks=get_tasks(
        tasks=MIEB_common_tasks
        + [
            "WITT2IRetrieval",
            "XFlickr30kCoT2IRetrieval",
            "XM3600T2IRetrieval",
            "VisualSTS17Eng",
            "VisualSTS-b-Eng",
            "VisualSTS17Multilingual",
            "VisualSTS-b-Multilingual",
        ],
    ),
    description="""MIEB(Multilingual) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 130 tasks and a total of 39 languages.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document understanding, visual STS, and CV-centric tasks. This benchmark consists of MIEB(eng) + 3 multilingual retrieval
    datasets + the multilingual parts of VisualSTS-b and VisualSTS-16.""",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  doi = {10.48550/ARXIV.2504.10471},
  journal = {arXiv preprint arXiv:2504.10471},
  publisher = {arXiv},
  title = {MIEB: Massive Image Embedding Benchmark},
  url = {https://arxiv.org/abs/2504.10471},
  year = {2025},
}
""",
)

MIEB_LITE = MIEBBenchmark(
    name="MIEB(lite)",
    display_name="Image-Text, Lite",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-map-landscape.svg",
    tasks=get_tasks(
        tasks=[
            # Image Classification
            "Country211",
            "DTD",
            "EuroSAT",
            "GTSRB",
            "OxfordPets",
            "PatchCamelyon",
            "RESISC45",
            "SUN397",
            # Clustering
            "ImageNetDog15Clustering",
            "TinyImageNetClustering",
            # ZeroShotClassification
            "CIFAR100ZeroShot",
            "Country211ZeroShot",
            "FER2013ZeroShot",
            "FGVCAircraftZeroShot",
            "Food101ZeroShot",
            "OxfordPetsZeroShot",
            "StanfordCarsZeroShot",
            # Any2AnyMultipleChoice
            "BLINKIT2IMultiChoice",
            "CVBenchCount",
            "CVBenchRelation",
            "CVBenchDepth",
            "CVBenchDistance",
            # ImageTextPairClassification
            "AROCocoOrder",
            "AROFlickrOrder",
            "AROVisualAttribution",
            "AROVisualRelation",
            "Winoground",
            "ImageCoDe",
            # VisualSTS
            "STS13VisualSTS",
            "STS15VisualSTS",
            "VisualSTS17Multilingual",
            "VisualSTS-b-Multilingual",
            # Any2AnyRetrieval
            "CIRRIT2IRetrieval",
            "CUB200I2IRetrieval",
            "Fashion200kI2TRetrieval",
            "HatefulMemesI2TRetrieval",
            "InfoSeekIT2TRetrieval",
            "NIGHTSI2IRetrieval",
            "OVENIT2TRetrieval",
            "RP2kI2IRetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VisualNewsI2TRetrieval",
            "VQA2IT2TRetrieval",
            "WebQAT2ITRetrieval",
            "WITT2IRetrieval",
            "XM3600T2IRetrieval",
        ],
    ),
    description="""MIEB(lite) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 51 tasks.
    This is a lite version of MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining
    relative rank of models.""",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  doi = {10.48550/ARXIV.2504.10471},
  journal = {arXiv preprint arXiv:2504.10471},
  publisher = {arXiv},
  title = {MIEB: Massive Image Embedding Benchmark},
  url = {https://arxiv.org/abs/2504.10471},
  year = {2025},
}
""",
)

MIEB_IMG = MIEBBenchmark(
    name="MIEB(Img)",
    display_name="Image only",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-pictures.svg",
    tasks=get_tasks(
        tasks=[
            "CUB200I2IRetrieval",
            "FORBI2IRetrieval",
            "GLDv2I2IRetrieval",
            "METI2IRetrieval",
            "NIGHTSI2IRetrieval",
            "ROxfordEasyI2IRetrieval",
            "ROxfordMediumI2IRetrieval",
            "ROxfordHardI2IRetrieval",
            "RP2kI2IRetrieval",
            "RParisEasyI2IRetrieval",
            "RParisMediumI2IRetrieval",
            "RParisHardI2IRetrieval",
            "SketchyI2IRetrieval",
            "SOPI2IRetrieval",
            "StanfordCarsI2IRetrieval",
            "Birdsnap",
            "Caltech101",
            "CIFAR10",
            "CIFAR100",
            "Country211",
            "DTD",
            "EuroSAT",
            "FER2013",
            "FGVCAircraft",
            "Food101Classification",
            "GTSRB",
            "Imagenet1k",
            "MNIST",
            "OxfordFlowersClassification",
            "OxfordPets",
            "PatchCamelyon",
            "RESISC45",
            "StanfordCars",
            "STL10",
            "SUN397",
            "UCF101",
            "CIFAR10Clustering",
            "CIFAR100Clustering",
            "ImageNetDog15Clustering",
            "ImageNet10Clustering",
            "TinyImageNetClustering",
            "VOC2007",
            "STS12VisualSTS",
            "STS13VisualSTS",
            "STS14VisualSTS",
            "STS15VisualSTS",
            "STS16VisualSTS",
            "STS17MultilingualVisualSTS",
            "STSBenchmarkMultilingualVisualSTS",
        ],
    ),
    description="A image-only version of MIEB(Multilingual) that consists of 49 tasks.",
    reference="https://arxiv.org/abs/2504.10471",
    citation=r"""
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  doi = {10.48550/ARXIV.2504.10471},
  journal = {arXiv preprint arXiv:2504.10471},
  publisher = {arXiv},
  title = {MIEB: Massive Image Embedding Benchmark},
  url = {https://arxiv.org/abs/2504.10471},
  year = {2025},
}
""",
    contacts=["gowitheflow-1998", "isaac-chung"],
)

BEIR_PL = Benchmark(
    name="BEIR-PL",
    tasks=get_tasks(
        languages=["pol"],
        tasks=[
            "MSMARCO-PL",
            "TRECCOVID-PL",
            "NFCorpus-PL",
            "NQ-PL",
            "HotpotQA-PL",
            "FiQA-PL",
            "ArguAna-PL",
            "Touche2020-PL",
            "CQADupstackRetrieval-PL",
            "Quora-PL",
            "DBPedia-PL",
            "SCIDOCS-PL",
            "SciFact-PL",
        ],
        eval_splits=["test"],
    ),
    description="BEIR-PL is a benchmark for evaluating text embedding models on Polish language data.",
    reference="https://arxiv.org/abs/2305.19840",
    citation=r"""
@misc{wojtasik2024beirplzeroshotinformation,
  archiveprefix = {arXiv},
  author = {Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
  eprint = {2305.19840},
  primaryclass = {cs.IR},
  title = {BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language},
  url = {https://arxiv.org/abs/2305.19840},
  year = {2024},
}
""",
)


BUILT_MTEB = Benchmark(
    name="BuiltBench(eng)",
    tasks=get_tasks(
        tasks=[
            "BuiltBenchClusteringP2P",
            "BuiltBenchClusteringS2S",
            "BuiltBenchRetrieval",
            "BuiltBenchReranking",
        ],
    ),
    description='"Built-Bench" is an ongoing effort aimed at evaluating text embedding models in the context of built asset management, spanning over various disciplines such as architecture, engineering, construction, and operations management of the built environment.',
    reference="https://arxiv.org/abs/2411.12056",
    citation=r"""
@article{shahinmoghadam2024benchmarking,
  author = {Shahinmoghadam, Mehrzad and Motamedi, Ali},
  journal = {arXiv preprint arXiv:2411.12056},
  title = {Benchmarking pre-trained text embedding models in aligning built asset information},
  year = {2024},
}
""",
    contacts=["mehrzadshm"],
)

ENCODECHKA = Benchmark(
    name="Encodechka",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                # PI
                "RUParaPhraserSTS",
                # SA
                "SentiRuEval2016",
                # TI
                "RuToxicOKMLCUPClassification",
                # IA
                "InappropriatenessClassificationv2",
                # IC, ICX
                "RuNLUIntentClassification",
            ]
        )
        +
        # NLI
        get_tasks(tasks=["XNLI"], eval_splits=["test"], languages=["rus-Cyrl"])
        # STS
        + get_tasks(
            tasks=["RuSTSBenchmarkSTS"],
            eval_splits=["validation"],
            languages=["rus-Cyrl"],
        ),
    ),
    description="A benchmark for evaluating text embedding models on Russian data.",
    reference="https://github.com/avidale/encodechka",
    citation=r"""
@misc{dale_encodechka,
  author = {Dale, David},
  editor = {habr.com},
  month = {June},
  note = {[Online; posted 12-June-2022]},
  title = {Russian rating of sentence encoders},
  url = {https://habr.com/ru/articles/669674/},
  year = {2022},
}
""",
)

VIDORE = Benchmark(
    name="ViDoRe(v1)",
    tasks=get_tasks(
        tasks=[
            "VidoreArxivQARetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VidoreSyntheticDocQAEnergyRetrieval",
            "VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        ],
    ),
    description="Retrieve associated pages according to questions.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{faysse2024colpali,
  author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  journal = {arXiv preprint arXiv:2407.01449},
  title = {ColPali: Efficient Document Retrieval with Vision Language Models},
  year = {2024},
}
""",
)

VIDORE_V2 = Benchmark(
    name="ViDoRe(v2)",
    tasks=get_tasks(
        tasks=[
            "Vidore2ESGReportsRetrieval",
            "Vidore2EconomicsReportsRetrieval",
            "Vidore2BioMedicalLecturesRetrieval",
            "Vidore2ESGReportsHLRetrieval",
        ],
    ),
    description="Retrieve associated pages according to questions.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
)

VIDORE_V3 = VidoreBenchmark(
    name="ViDoRe(v3)",
    display_name="ViDoRe V3",
    language_view=[
        "deu-Latn",
        "eng-Latn",
        "fra-Latn",
        "ita-Latn",
        "por-Latn",
        "spa-Latn",
    ],
    icon="https://cdn-uploads.huggingface.co/production/uploads/66e16a677c2eb2da5109fb5c/x99xqw__fl2UaPbiIdC_f.png",
    tasks=get_tasks(
        tasks=[
            "Vidore3FinanceEnRetrieval",
            "Vidore3IndustrialRetrieval",
            "Vidore3ComputerScienceRetrieval",
            "Vidore3PharmaceuticalsRetrieval",
            "Vidore3HrRetrieval",
            "Vidore3FinanceFrRetrieval",
            "Vidore3PhysicsRetrieval",
            "Vidore3EnergyRetrieval",
            "Vidore3TelecomRetrieval",
            "Vidore3NuclearRetrieval",
        ]
    ),
    description="ViDoRe V3 sets a new industry gold standard for multi-modal, enterprise document visual retrieval evaluation. It addresses a critical challenge in production RAG systems: retrieving accurate information from complex, visually-rich documents. The benchmark includes both open and closed datasets: to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml).",
    reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
    citation=r"""
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
)

VISUAL_DOCUMENT_RETRIEVAL = VidoreBenchmark(
    name="ViDoRe(v1&v2)",
    aliases=["VisualDocumentRetrieval"],
    display_name="ViDoRe (V1&V2)",
    tasks=get_tasks(
        tasks=[
            # v1
            "VidoreArxivQARetrieval",
            "VidoreDocVQARetrieval",
            "VidoreInfoVQARetrieval",
            "VidoreTabfquadRetrieval",
            "VidoreTatdqaRetrieval",
            "VidoreShiftProjectRetrieval",
            "VidoreSyntheticDocQAAIRetrieval",
            "VidoreSyntheticDocQAEnergyRetrieval",
            "VidoreSyntheticDocQAGovernmentReportsRetrieval",
            "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
            # v2
            "Vidore2ESGReportsRetrieval",
            "Vidore2EconomicsReportsRetrieval",
            "Vidore2BioMedicalLecturesRetrieval",
            "Vidore2ESGReportsHLRetrieval",
        ],
    ),
    description="A benchmark for evaluating visual document retrieval, combining ViDoRe v1 and v2.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
)

R2MED = Benchmark(
    name="R2MED",
    display_name="Reasoning-driven medical retrieval",
    tasks=get_tasks(
        tasks=[
            "R2MEDBiologyRetrieval",
            "R2MEDBioinformaticsRetrieval",
            "R2MEDMedicalSciencesRetrieval",
            "R2MEDMedXpertQAExamRetrieval",
            "R2MEDMedQADiagRetrieval",
            "R2MEDPMCTreatmentRetrieval",
            "R2MEDPMCClinicalRetrieval",
            "R2MEDIIYiClinicalRetrieval",
        ]
    ),
    description="""R2MED: First Reasoning-Driven Medical Retrieval Benchmark.
    R2MED is a high-quality, high-resolution information retrieval (IR) dataset designed for medical scenarios.
    It contains 876 queries with three retrieval tasks, five medical scenarios, and twelve body systems.
    """,
    reference="https://r2med.github.io/",
    citation=r"""
@article{li2025r2med,
  author = {Li, Lei and Zhou, Xiao and Liu, Zheng},
  journal = {arXiv preprint arXiv:2505.14558},
  title = {R2MED: A Benchmark for Reasoning-Driven Medical Retrieval},
  year = {2025},
}
""",
)


VN_MTEB = Benchmark(
    name="VN-MTEB (vie, v1)",
    display_name="Vietnamese",
    icon="https://raw.githubusercontent.com/lipis/flag-icons/refs/heads/main/flags/4x3/vn.svg",
    tasks=get_tasks(
        languages=["vie"],
        exclusive_language_filter=True,
        tasks=[
            # Retrieval
            "ArguAna-VN",
            "SciFact-VN",
            "ClimateFEVER-VN",
            "FEVER-VN",
            "DBPedia-VN",
            "NQ-VN",
            "HotpotQA-VN",
            "MSMARCO-VN",
            "TRECCOVID-VN",
            "FiQA2018-VN",
            "NFCorpus-VN",
            "SCIDOCS-VN",
            "Touche2020-VN",
            "Quora-VN",
            "CQADupstackAndroid-VN",
            "CQADupstackGis-VN",
            "CQADupstackMathematica-VN",
            "CQADupstackPhysics-VN",
            "CQADupstackProgrammers-VN",
            "CQADupstackStats-VN",
            "CQADupstackTex-VN",
            "CQADupstackUnix-VN",
            "CQADupstackWebmasters-VN",
            "CQADupstackWordpress-VN",
            # Classification
            "Banking77VNClassification",
            "EmotionVNClassification",
            "AmazonCounterfactualVNClassification",
            "MTOPDomainVNClassification",
            "TweetSentimentExtractionVNClassification",
            "ToxicConversationsVNClassification",
            "ImdbVNClassification",
            "MTOPIntentVNClassification",
            "MassiveScenarioVNClassification",
            "MassiveIntentVNClassification",
            "AmazonReviewsVNClassification",
            "AmazonPolarityVNClassification",
            # Pair Classification
            "SprintDuplicateQuestions-VN",
            "TwitterSemEval2015-VN",
            "TwitterURLCorpus-VN",
            # Clustering
            "TwentyNewsgroupsClustering-VN",
            "RedditClusteringP2P-VN",
            "StackExchangeClusteringP2P-VN",
            "StackExchangeClustering-VN",
            "RedditClustering-VN",
            # Reranking
            "SciDocsRR-VN",
            "AskUbuntuDupQuestions-VN",
            "StackOverflowDupQuestions-VN",
            # STS
            "BIOSSES-VN",
            "SICK-R-VN",
            "STSBenchmark-VN",
        ],
    ),
    description="A benchmark for text-embedding performance in Vietnamese.",
    reference="https://arxiv.org/abs/2507.21500",
    citation=r"""
@misc{pham2025vnmtebvietnamesemassivetext,
  archiveprefix = {arXiv},
  author = {Loc Pham and Tung Luu and Thu Vo and Minh Nguyen and Viet Hoang},
  eprint = {2507.21500},
  primaryclass = {cs.CL},
  title = {VN-MTEB: Vietnamese Massive Text Embedding Benchmark},
  url = {https://arxiv.org/abs/2507.21500},
  year = {2025},
}
""",
    contacts=["BaoLocPham"],
)

JINA_VDR = Benchmark(
    name="JinaVDR",
    display_name="Jina Visual Document Retrieval",
    tasks=get_tasks(
        tasks=[
            "JinaVDRMedicalPrescriptionsRetrieval",
            "JinaVDRStanfordSlideRetrieval",
            "JinaVDRDonutVQAISynHMPRetrieval",
            "JinaVDRTableVQARetrieval",
            "JinaVDRChartQARetrieval",
            "JinaVDRTQARetrieval",
            "JinaVDROpenAINewsRetrieval",
            "JinaVDREuropeanaDeNewsRetrieval",
            "JinaVDREuropeanaEsNewsRetrieval",
            "JinaVDREuropeanaItScansRetrieval",
            "JinaVDREuropeanaNlLegalRetrieval",
            "JinaVDRHindiGovVQARetrieval",
            "JinaVDRAutomobileCatelogRetrieval",
            "JinaVDRBeveragesCatalogueRetrieval",
            "JinaVDRRamensBenchmarkRetrieval",
            "JinaVDRJDocQARetrieval",
            "JinaVDRHungarianDocQARetrieval",
            "JinaVDRArabicChartQARetrieval",
            "JinaVDRArabicInfographicsVQARetrieval",
            "JinaVDROWIDChartsRetrieval",
            "JinaVDRMPMQARetrieval",
            "JinaVDRJina2024YearlyBookRetrieval",
            "JinaVDRWikimediaCommonsMapsRetrieval",
            "JinaVDRPlotQARetrieval",
            "JinaVDRMMTabRetrieval",
            "JinaVDRCharXivOCRRetrieval",
            "JinaVDRStudentEnrollmentSyntheticRetrieval",
            "JinaVDRGitHubReadmeRetrieval",
            "JinaVDRTweetStockSyntheticsRetrieval",
            "JinaVDRAirbnbSyntheticRetrieval",
            "JinaVDRShanghaiMasterPlanRetrieval",
            "JinaVDRWikimediaCommonsDocumentsRetrieval",
            "JinaVDREuropeanaFrNewsRetrieval",
            "JinaVDRDocQAHealthcareIndustryRetrieval",
            "JinaVDRDocQAAI",
            "JinaVDRShiftProjectRetrieval",
            "JinaVDRTatQARetrieval",
            "JinaVDRInfovqaRetrieval",
            "JinaVDRDocVQARetrieval",
            "JinaVDRDocQAGovReportRetrieval",
            "JinaVDRTabFQuadRetrieval",
            "JinaVDRDocQAEnergyRetrieval",
            "JinaVDRArxivQARetrieval",
        ],
    ),
    description="Multilingual, domain-diverse and layout-rich document retrieval benchmark.",
    reference="https://arxiv.org/abs/2506.18902",
    citation=r"""@misc{günther2025jinaembeddingsv4universalembeddingsmultimodal,
  archiveprefix = {arXiv},
  author = {Michael Günther and Saba Sturua and Mohammad Kalim Akram and Isabelle Mohr and Andrei Ungureanu and Bo Wang and Sedigheh Eslami and Scott Martens and Maximilian Werk and Nan Wang and Han Xiao},
  eprint = {2506.18902},
  primaryclass = {cs.AI},
  title = {jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval},
  url = {https://arxiv.org/abs/2506.18902},
  year = {2025},
}""",
)


HUME = HUMEBenchmark(
    name="HUME(v1)",
    display_name="Human Benchmark",
    # icon="https://raw.githubusercontent.com/huggingface/benchmarks/main/benchmarks/assets/hume.png",
    tasks=get_tasks(
        tasks=[
            "HUMEEmotionClassification",
            "HUMEToxicConversationsClassification",
            "HUMETweetSentimentExtractionClassification",
            "HUMEMultilingualSentimentClassification",
            "HUMEArxivClusteringP2P",
            "HUMERedditClusteringP2P",
            "HUMEWikiCitiesClustering",
            "HUMESIB200ClusteringS2S",
            "HUMECore17InstructionReranking",
            "HUMENews21InstructionReranking",
            "HUMERobust04InstructionReranking",
            "HUMEWikipediaRerankingMultilingual",
            "HUMESICK-R",
            "HUMESTS12",
            "HUMESTSBenchmark",
            "HUMESTS22",
        ],
        languages=[
            "eng-Latn",
            "ara-Arab",
            "rus-Cyrl",
            "dan-Latn",
            "nob-Latn",
        ],
    ),
    description="The HUME benchmark is designed to evaluate the performance of text embedding models and humans on a comparable set of tasks. This captures areas where models perform better than human annotators and the reverse. In the paper, we go further into the analysis and what conclusions can be drawn.",
    reference=None,
    citation=None,
    contacts=["AdnanElAssadi56", "KennethEnevoldsen", "isaac-chung", "Samoed"],
)

JMTEB_V2 = Benchmark(
    name="JMTEB(v2)",
    display_name="Japanese",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/jp.svg",
    tasks=get_tasks(
        languages=["jpn"],
        tasks=[
            # Clustering (3)
            "LivedoorNewsClustering.v2",
            "MewsC16JaClustering",
            "SIB200ClusteringS2S",
            # Classification (7)
            "AmazonReviewsClassification",
            "AmazonCounterfactualClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "JapaneseSentimentClassification",
            "SIB200Classification",
            "WRIMEClassification",
            # STS (2)
            "JSTS",
            "JSICK",
            # Retrieval (11)
            "JaqketRetrieval",
            "MrTidyRetrieval",
            "JaGovFaqsRetrieval",
            "NLPJournalTitleAbsRetrieval.V2",
            "NLPJournalTitleIntroRetrieval.V2",
            "NLPJournalAbsIntroRetrieval.V2",
            "NLPJournalAbsArticleRetrieval.V2",
            "JaCWIRRetrieval",
            "MIRACLRetrieval",
            "MintakaRetrieval",
            "MultiLongDocRetrieval",
            # Reranking (5)
            "ESCIReranking",
            "JQaRAReranking",
            "JaCWIRReranking",
            "MIRACLReranking",
            "MultiLongDocReranking",
        ],
    ),
    description="JMTEB is a benchmark for evaluating Japanese text embedding models. In v2, we have extended the benchmark to 28 datasets, enabling more comprehensive evaluation compared with v1 (MTEB(jpn, v1)).",
    reference="https://github.com/sbintuitions/JMTEB",
    citation=r"""
@article{li2025jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  issue = {3},
  journal = {Vol.2025-NL-265,No.3,1-15},
  month = {sep},
  title = {{JMTEB and JMTEB-lite: Japanese Massive Text Embedding Benchmark and Its Lightweight Version}},
  year = {2025},
}
""",
    contacts=["lsz05"],
)

JMTEB_LITE_V1 = Benchmark(
    name="JMTEB-lite(v1)",
    display_name="Japanese",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/jp.svg",
    tasks=get_tasks(
        languages=["jpn"],
        tasks=[
            # Clustering (3)
            "LivedoorNewsClustering.v2",
            "MewsC16JaClustering",
            "SIB200ClusteringS2S",
            # Classification (7)
            "AmazonReviewsClassification",
            "AmazonCounterfactualClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "JapaneseSentimentClassification",
            "SIB200Classification",
            "WRIMEClassification",
            # STS (2)
            "JSTS",
            "JSICK",
            # Retrieval (11)
            "JaqketRetrievalLite",
            "MrTyDiJaRetrievalLite",
            "JaGovFaqsRetrieval",
            "NLPJournalTitleAbsRetrieval.V2",
            "NLPJournalTitleIntroRetrieval.V2",
            "NLPJournalAbsIntroRetrieval.V2",
            "NLPJournalAbsArticleRetrieval.V2",
            "JaCWIRRetrievalLite",
            "MIRACLJaRetrievalLite",
            "MintakaRetrieval",
            "MultiLongDocRetrieval",
            # Reranking (5)
            "ESCIReranking",
            "JQaRARerankingLite",
            "JaCWIRRerankingLite",
            "MIRACLReranking",
            "MultiLongDocReranking",
        ],
    ),
    description="JMTEB-lite is a lightweight version of JMTEB. It makes agile evaluation possible by reaching an average of 5x faster evaluation comparing with JMTEB, as 6 heavy datasets in JMTEB are optimized with hard negative pooling strategy, making them much smaller. The result of JMTEB-lite is proved to be highly relevant with that of JMTEB, making it a faithful preview of JMTEB.",
    reference="https://huggingface.co/datasets/sbintuitions/JMTEB-lite",
    citation=r"""
@article{li2025jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  issue = {3},
  journal = {Vol.2025-NL-265,No.3,1-15},
  month = {sep},
  title = {{JMTEB and JMTEB-lite: Japanese Massive Text Embedding Benchmark and Its Lightweight Version}},
  year = {2025},
}
""",
    contacts=["lsz05"],
)
