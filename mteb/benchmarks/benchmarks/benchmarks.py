from mteb.benchmarks.benchmark import (
    Benchmark,
    BenchmarkAggregation,
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
    description="English text embedding quality across classification, clustering, retrieval, reranking, pair classification, and semantic similarity, prioritizing tasks not commonly used for fine-tuning to give a more realistic estimate of generalization performance. The original v1 leaderboard is available under [MTEB(eng, v1)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v1%29).",
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
    description="English text embedding quality across classification, clustering, retrieval, reranking, pair classification, and semantic similarity. We recommend using [MTEB(eng, v2)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28eng%2C+v2%29) instead, which resolves [a known scoring bug](https://github.com/embeddings-benchmark/mteb/issues/1156), uses updated task versions, and removes common fine-tuning datasets such as MSMARCO for more comparable scores.",
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
    superseded_by=["MTEB(eng, v2)"],
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
    description="Russian text embedding quality across classification, clustering, reranking, pair classification, retrieval, and semantic similarity, including novel Russian-specific tasks in each category.",
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
    superseded_by=["MTEB(rus, v1.1)"],
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
    description="Russian text embedding quality across classification, clustering, reranking, pair classification, retrieval, and semantic similarity. In v1.1, MIRACLRetrieval and RiaNewsRetrieval were replaced with their HardNegatives variants (v2), which include improved default prompts.",
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
    description="Scientific text embedding quality in Russian and English across bitext mining, classification, retrieval, and regression tasks, using data sourced from eLibrary, Russia's largest electronic library of scientific publications.",
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
    description="Instruction-following retrieval quality, measuring how well models retrieve relevant documents when given detailed natural language instructions alongside queries.",
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
    description="Instruction-following retrieval quality in expert domains including healthcare, finance, and legal, where queries are paired with domain-specific natural language instructions.",
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
    description="Legal document retrieval across case documents, statutes, legal Q&A, and legal summarization in multiple languages.",
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
    description="Medical information retrieval across clinical, biomedical, and consumer health domains, spanning retrieval, reranking, and clustering tasks.",
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
    description="Multilingual bitext mining quality across diverse language pairs, drawn from the MINERS benchmark for evaluating semantic retrieval in multilingual settings.",
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
    description="Scandinavian text embedding quality covering Danish, Swedish, Norwegian Bokmål, and Nynorsk and spanning classification, clustering, retrieval as well as bitext  tasks across dialects or written forms.",
    reference="https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/",
    citation=r"""
@article{enevoldsenScandinavianEmbeddingBenchmarks2024,
  author = {Enevoldsen, Kenneth and Kardos, Márton and Muennighoff, Niklas and Nielbo, Kristoffer},
  language = {en},
  month = feb,
  shorttitle = {The {Scandinavian} {Embedding} {Benchmarks}},
  title = {The {Scandinavian} {Embedding} {Benchmarks}: {Comprehensive} {Assessment} of {Multilingual} and {Monolingual} {Text} {Embedding}},
  url = {https://openreview.net/forum?id=pJl_i7HIA72},
  urldate = {2024-04-12},
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
    description="Code information retrieval across diverse programming languages and coding tasks, including code search, question answering, and text-to-SQL retrieval.",
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
    description="Reasoning capabilities of retrieval models, framing commonsense, temporal, and domain-specific reasoning tasks as retrieval problems.",
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
    description="French text embedding quality across classification, clustering, pair classification, reranking, retrieval, and semantic similarity, using high-quality native French datasets.",
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
    description="German text embedding quality across classification, clustering, pair classification, reranking, retrieval, and semantic similarity.",
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
        tasks=[
            # @KennethEnevoldsen: We could probably expand this to a more solid benchmark, but for now I have left it as is.
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
    description="Korean text embedding quality across classification, reranking, retrieval, and semantic similarity.",
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
    description="Polish text embedding quality across classification, clustering, pair classification, retrieval, and semantic similarity, combining adapted community datasets with a novel Polish scientific literature corpus (PLSC).",
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

MTEB_PT = Benchmark(
    name="MTEB(por, v1)",
    aliases=["MTEB(por)"],
    display_name="Portuguese",
    icon="https://raw.githubusercontent.com/lipis/flag-icons/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/br.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["por"],
            tasks=[
                # STS (3)
                "Assin2STS",
                "SICK-BR-STS",
                "STSBenchmarkMultilingualSTS",
                # Classification (3)
                "MassiveIntentClassification",
                "BrazilianToxicTweetsClassification",
                "HateSpeechPortugueseClassification",
                # Reranking (3)
                "MultiLongDocReranking",
                "WikipediaRerankingMultilingual",
                "XGlueWPRReranking",
                # Retrieval (1)
                "MultiLongDocRetrieval",
                "WikipediaRetrievalMultilingual",
            ],
        )
        # Classification (2)
        + (
            get_task(
                "MultiHateClassification", eval_splits=["test"], hf_subsets=["por"]
            ),
        )
        + (
            get_task(
                "TweetSentimentClassification",
                eval_splits=["test"],
                hf_subsets=["portuguese"],
            ),
        )
        # Retrieval (1)
        + (get_task("WebFAQRetrieval", eval_splits=["test"], hf_subsets=["por"]),)
    ),
    description="Portuguese text embedding quality benchmark across semantic text similarity, classification, reranking and retrieval.",
    reference=None,
    citation=r"""
@misc{okamura2026multilingualaveragesmtebptbenchmark,
  archiveprefix = {arXiv},
  author = {Lucas Hideki Takeuchi Okamura and Alexandre Alcoforado and Anna Helena Reali Costa},
  eprint = {2607.04071},
  primaryclass = {cs.CL},
  title = {Beyond Multilingual Averages: MTEB-PT, a Benchmark for Portuguese Sentence Encoders},
  url = {https://arxiv.org/abs/2607.04071},
  year = {2026},
}
""",
    contacts=["Lucas-Okamura"],
)

MTEB_SPA = Benchmark(
    name="MTEB(spa, v1)",
    aliases=["MTEB(spa)"],
    display_name="Spanish",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/es.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["spa"],
            tasks=[
                # Classification
                "SpanishNewsClassification.v2",
                "SpanishSentimentClassification.v2",
                # Clustering
                "MLSUMClusteringP2P",
                "MLSUMClusteringS2S",
                # Pair Classification
                "PawsXPairClassification",
                "XNLI",
                # Reranking
                "MIRACLReranking",
                # Retrieval
                "MIRACLRetrievalHardNegatives.v2",
                "MintakaRetrieval",
                "SpanishPassageRetrievalS2P",
                "SpanishPassageRetrievalS2S",
                "XPQARetrieval",
                # STS
                "STSES",
                "STSBenchmarkMultilingualSTS",
                "STS17",
            ],
        )
        + (get_task("STS22", eval_splits=["test"], hf_subsets=["es"]),)
    ),
    description="Spanish text embedding quality across classification, clustering, pair classification, reranking, retrieval, and semantic similarity. For discussion on benchmark construction, see the [original submission](https://github.com/embeddings-benchmark/mteb/pull/4053).",
    reference=None,
    citation=None,
    contacts=["Clemente-H"],
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
    description="Code retrieval quality across a wide range of popular programming languages, covering code search, text-to-SQL, and code feedback tasks.",
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
    description="Multilingual text embedding quality across 250+ languages spanning bitext mining, classification, clustering, retrieval, reranking, and semantic similarity. Superseded by MTEB(Multilingual, v2) after SNLHierarchicalClustering was removed from Hugging Face Hub.",
    reference="https://arxiv.org/abs/2502.13595",
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
    superseded_by=["MTEB(Multilingual, v2)"],
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
    description="MMTEB measures multilingual text embedding quality across 250+ languages spanning classification, clustering, retrieval semantic similarity and more, driven by curated community contributions.",
    reference="https://arxiv.org/abs/2502.13595",
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
    benchmark_hf_repo="mteb/MMTEB-Multilingual-v2",
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
    description="Japanese text embedding quality across clustering, classification, semantic similarity, pair classification, retrieval, and reranking.",
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
    language_view="all",
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
    description="Text embedding quality across Indic languages spanning bitext mining, classification, clustering, pair classification, retrieval, reranking, and semantic similarity.",
    reference=None,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)

afri_languages = [
    "aeb",
    "afr",
    "aka",
    "amh",
    "arb",
    "arq",
    "ary",
    "arz",
    "bam",
    "bem",
    "cjk",
    "dik",
    "dyu",
    "eng",
    "ewe",
    "fon",
    "fuv",
    "gaz",
    "hau",
    "ibo",
    "kab",
    "kam",
    "kbp",
    "kea",
    "kik",
    "kin",
    "kmb",
    "knc",
    "kon",
    "lin",
    "lua",
    "lug",
    "luo",
    "mos",
    "nbl",
    "nqo",
    "nso",
    "nus",
    "nya",
    "pcm",
    "plt",
    "por",
    "run",
    "sag",
    "sna",
    "som",
    "sot",
    "ssw",
    "swa",
    "swh",
    "taq",
    "tir",
    "tsn",
    "tso",
    "tum",
    "twi",
    "tzm",
    "umb",
    "ven",
    "wol",
    "xho",
    "yor",
    "zul",
]

afri_languages_lite = [
    "amh",
    "gaz",
    "ibo",
    "yor",
    "hau",
    "swh",
    "kin",
    "xho",
    "zul",
]

MTEB_AFRICA = Benchmark(
    name="MTEB(Africa, v1)",
    aliases=["AfriMTEB"],
    display_name="African",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-globe.svg",
    tasks=get_tasks(
        tasks=[
            # Original tasks
            "SIB200Classification",
            "FloresBitextMining",
            "SIB200ClusteringS2S",
            "NTREXBitextMining",
            "AfriXNLI",
            "EmotionAnalysisPlus",
            "AfriSentiClassification",
            "MasakhaNEWSClassification",
            "AfriHateClassification",
            "KinNewsClassification",
            "InjongoIntent",
            "SIB200Classification.v2",
            # Additional comprehensive tasks
            # Retrieval tasks
            "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives",
            "MrTidyRetrieval",
            "XQuADRetrieval",
            "XM3600T2IRetrieval",
            # Additional classification tasks
            "AfriSentiLangClassification",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            "MultilingualSentimentClassification",
            "NaijaSenti",
            "SouthAfricanLangClassification",
            "SiswatiNewsClassification",
            "SwahiliNewsClassification",
            "TswanaNewsClassification",
            "IsiZuluNewsClassification",
            # Additional bitext mining tasks
            "BibleNLPBitextMining",
            "NollySentiBitextMining",
            "Tatoeba",
            # Additional clustering tasks
            "MasakhaNEWSClusteringP2P",
            "MasakhaNEWSClusteringS2S",
            # Additional other tasks
            "XNLI",
            "MIRACLReranking",
            "SemRel24STS",
        ],
        languages=afri_languages,
        exclusive_language_filter=True,
    ),
    description="Text embedding quality across African languages spanning bitext mining, classification, clustering, pair classification, retrieval, reranking, and semantic similarity.",
    reference="https://arxiv.org/abs/2510.23896",
    citation=MMTEB_CITATION,
    contacts=["Kosei1227"],
)

MTEB_AFRICA_LITE = Benchmark(
    name="MTEB(Africa, v1, lite)",
    aliases=["AfriMTEB-Lite"],
    display_name="African Lite",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-globe.svg",
    tasks=get_tasks(
        tasks=[
            "AfriHateClassification",
            "AfriSentiClassification",
            "MasakhaNEWSClassification",
            "KinNewsClassification",
            "AfriXNLI",
            "EmotionAnalysisPlus",
            "FloresBitextMining",
            "InjongoIntent",
            "NTREXBitextMining",
            "SIB200Classification.v2",
            "SIB200Classification",
            "SIB200ClusteringS2S",
            "BelebeleRetrieval",
        ],
        languages=afri_languages_lite,
        exclusive_language_filter=True,
    ),
    description="Text embedding quality across 9 geographically diverse African languages, covering bitext mining, classification, clustering, pair classification, and retrieval. A computationally lighter subset of MTEB(Africa, v1) designed for faster evaluation.",
    reference="https://arxiv.org/abs/2510.23896",
    citation=MMTEB_CITATION,
    contacts=["Kosei1227"],
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
    language_view="all",
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
    description="Text embedding quality across European languages spanning bitext mining, classification, clustering, pair classification, retrieval, reranking, and semantic similarity.",
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
    description="Long-context retrieval quality across synthetic and real-world tasks featuring documents of varying length with dispersed target information.",
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

_LMEB_REFERENCE = "https://arxiv.org/abs/2603.12572"
_LMEB_CITATION = r"""
@misc{zhao2026lmeb,
  archiveprefix = {arXiv},
  author = {Zhao, Xinping and Hu, Xinshuo and Xu, Jiaxin and Tang, Danyu and Zhang, Xin and Zhou, Mengjia and Zhong, Yan and Zhou, Yao and Shan, Zifei and Zhang, Meishan and Hu, Baotian and Zhang, Min},
  eprint = {2603.12572},
  primaryclass = {cs.CL},
  title = {LMEB: Long-horizon Memory Embedding Benchmark},
  url = {https://arxiv.org/abs/2603.12572},
  year = {2026},
}
"""
_LMEB_EPISODIC_TASKS = [
    "EPBench",
    "KnowMeBench",
]
_LMEB_DIALOGUE_TASKS = [
    "LoCoMo",
    "LongMemEval",
    "REALTALK",
    "TMD",
    "MemBench",
    "ConvoMem",
]
_LMEB_SEMANTIC_TASKS = [
    "QASPER",
    "NovelQA",
    "PeerQA",
    "CovidQA",
    "ESGReports",
    "LMEBMLDR",
    "LooGLE",
    "LMEB_SciFact",
]
_LMEB_PROCEDURAL_TASKS = [
    "Gorilla",
    "ToolBench",
    "ReMe",
    "ProceduralMemBench",
    "MemGovern",
    "DeepPlanning",
]

LMEB = Benchmark(
    name="LMEB",
    display_name="Long-Horizon Memory",
    tasks=get_tasks(
        tasks=[
            *_LMEB_EPISODIC_TASKS,
            *_LMEB_DIALOGUE_TASKS,
            *_LMEB_SEMANTIC_TASKS,
            *_LMEB_PROCEDURAL_TASKS,
        ]
    ),
    description="Long-horizon memory retrieval quality across episodic, dialogue, semantic, and procedural retrieval tasks, measuring how well embedding models retrieve evidence in long-term memory scenarios.",
    reference=_LMEB_REFERENCE,
    citation=_LMEB_CITATION,
)

LMEB_EPISODIC = Benchmark(
    name="LMEB-Episodic",
    display_name="LMEB Episodic Memory",
    tasks=get_tasks(tasks=_LMEB_EPISODIC_TASKS),
    description="Episodic memory retrieval aims to recall past events grounded in temporal cues, entities, contents, and spatial context.",
    reference=_LMEB_REFERENCE,
    citation=_LMEB_CITATION,
)

LMEB_DIALOGUE = Benchmark(
    name="LMEB-Dialogue",
    display_name="LMEB Dialogue Memory",
    tasks=get_tasks(tasks=_LMEB_DIALOGUE_TASKS),
    description="Dialogue memory retrieval aims to maintain context across multi-turn interactions by recalling relevant dialogue history and user preference.",
    reference=_LMEB_REFERENCE,
    citation=_LMEB_CITATION,
)

LMEB_SEMANTIC = Benchmark(
    name="LMEB-Semantic",
    display_name="LMEB Semantic Memory",
    tasks=get_tasks(tasks=_LMEB_SEMANTIC_TASKS),
    description="Semantic memory retrieval focuses on recalling general knowledge and concepts that are largely independent of time or specific context.",
    reference=_LMEB_REFERENCE,
    citation=_LMEB_CITATION,
)

LMEB_PROCEDURAL = Benchmark(
    name="LMEB-Procedural",
    display_name="LMEB Procedural Memory",
    tasks=get_tasks(tasks=_LMEB_PROCEDURAL_TASKS),
    description="Procedural memory retrieval focuses on recalling learned skills, action patterns, and structured procedures that guide task execution and multi-step reasoning.",
    reference=_LMEB_REFERENCE,
    citation=_LMEB_CITATION,
)

BRIGHT = Benchmark(
    name="BRIGHT",
    display_name="Reasoning Retrieval",
    tasks=get_tasks(tasks=["BrightRetrieval"], eval_splits=["standard"]),
    description="Reasoning-intensive retrieval quality across real-world queries spanning diverse domains including economics, psychology, mathematics, and coding, drawn from naturally occurring and carefully curated human data.",
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
    superseded_by=["BRIGHT(v1.1)"],
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
    description="Reasoning-intensive retrieval quality across real-world queries spanning diverse domains, filtered to longer documents to stress-test models on extended contexts.",
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
    superseded_by=["BRIGHT(v1.1)"],
)

BRIGHT_V1_1 = Benchmark(
    name="BRIGHT(v1.1)",
    display_name="Reasoning Retrieval",
    tasks=get_tasks(
        tasks=[
            "BrightBiologyRetrieval",
            "BrightEarthScienceRetrieval",
            "BrightEconomicsRetrieval",
            "BrightPsychologyRetrieval",
            "BrightRoboticsRetrieval",
            "BrightStackoverflowRetrieval",
            "BrightSustainableLivingRetrieval",
            "BrightPonyRetrieval",
            "BrightLeetcodeRetrieval",
            "BrightAopsRetrieval",
            "BrightTheoremQATheoremsRetrieval",
            "BrightTheoremQAQuestionsRetrieval",
            "BrightBiologyLongRetrieval",
            "BrightEarthScienceLongRetrieval",
            "BrightEconomicsLongRetrieval",
            "BrightPsychologyLongRetrieval",
            "BrightRoboticsLongRetrieval",
            "BrightStackoverflowLongRetrieval",
            "BrightSustainableLivingLongRetrieval",
            "BrightPonyLongRetrieval",
        ],
    ),
    description="Reasoning-intensive retrieval quality across real-world queries spanning diverse domains including economics, psychology, mathematics, and coding. v1.1 restructures tasks into separate datasets and adds per-task prompts.",
    reference="https://brightbenchmark.github.io/",
    citation=r"""
@article{su2024bright,
  author = {Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
  journal = {arXiv preprint arXiv:2407.12883},
  title = {Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
  year = {2024},
}
""",
    benchmark_hf_repo="mteb/BRIGHT",
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
    description="Code retrieval quality for retrieval-augmented generation, covering programming solutions, online tutorials, library documentation, and Stack Overflow posts.",
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
    description="Zero-shot retrieval quality across a heterogeneous set of IR tasks and domains, providing a common framework for comparing NLP-based retrieval models.",
    reference="https://arxiv.org/abs/2104.08663",
    citation=r"""
@inproceedings{thakur2021beir,
  author = {Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
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
    description="Zero-shot retrieval quality using subsets of the BEIR datasets, designed for faster evaluation with reduced computational cost.",
    reference="https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6",
    citation=None,
)

NANOBEIR_EXTENDED = Benchmark(
    name="NanoBEIR-multilingual",
    display_name="Multilingual NanoBEIR",
    tasks=get_tasks(
        tasks=[
            "MultilingualNanoArguAnaRetrieval",
            "MultilingualNanoClimateFeverRetrieval",
            "MultilingualNanoDBPediaRetrieval",
            "MultilingualNanoFEVERRetrieval",
            "MultilingualNanoFiQA2018Retrieval",
            "MultilingualNanoHotpotQARetrieval",
            "MultilingualNanoMSMARCORetrieval",
            "MultilingualNanoNFCorpusRetrieval",
            "MultilingualNanoNQRetrieval",
            "MultilingualNanoQuoraRetrieval",
            "MultilingualNanoSCIDOCSRetrieval",
            "MultilingualNanoSciFactRetrieval",
            "MultilingualNanoTouche2020Retrieval",
        ],
    ),
    description="A translated benchmark targeting zero-shot retrieval quality using translated subsets of the BEIR datasets, designed for faster evaluation during training with reduced computational cost.",
    reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
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
                "MultilingualSentiment",
            ],
        )
        + get_tasks(
            tasks=[
                "ATEC",
                "BQ",
                "STSB",
            ],
            eval_splits=["test"],
        )
    ),
    description="Chinese text embedding quality across retrieval, reranking, pair classification, clustering, classification, and semantic similarity.",
    reference="https://github.com/FlagOpen/FlagEmbedding/tree/master/research/C_MTEB",
    citation=r"""
@misc{xiao2024cpackpackagedresourcesadvance,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  url = {https://arxiv.org/abs/2309.07597},
  year = {2024},
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
    description="Persian text embedding quality across classification, clustering, pair classification, reranking, retrieval, semantic similarity, and summarization retrieval.",
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
    description="Persian text embedding quality across classification, clustering, pair classification, reranking, retrieval, semantic similarity, and summarization retrieval. In v2, large datasets were optimized for accessibility, low-quality datasets were removed, and higher-quality data was added; see the [main PR](https://github.com/embeddings-benchmark/mteb/pull/3157) for details.",
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
    aliases=["ChemTEB(v1)"],
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
    description="Chemical domain text embedding quality across bitext mining, classification, clustering, pair classification, and retrieval.",
    reference="https://arxiv.org/abs/2412.00532",
    citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}
""",
)

CHEMTEB_V1_1 = Benchmark(
    name="ChemTEB(v1.1)",
    aliases=["ChemTEB(latest)"],
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
            "ChemRxivRetrieval",
        ],
    ),
    description="Chemical domain text embedding quality across bitext mining, classification, clustering, pair classification, and retrieval. v1.1 adds the ChemRxivRetrieval task.",
    reference="https://arxiv.org/abs/2412.00532",
    citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
  year = {2024},
}

@article{kasmaee2025chembed,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Astaraki, Mahdi and Saloot, Mohammad Arshi and Sherck, Nicholas and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2508.01643},
  title = {Chembed: Enhancing chemical literature search through domain-specific text embeddings},
  year = {2025},
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
    description="Zero-shot retrieval quality in Dutch across the BEIR task suite, created through automated translation of the original English benchmark.",
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
    description="Dutch text embedding quality across classification, clustering, pair classification, multilabel classification, reranking, retrieval, and semantic similarity.",
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
    # MultiLabelClassification
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
    description="English image embedding quality across image classification (zero-shot and linear probing), clustering, retrieval, compositionality evaluation, document understanding, visual STS, and CV-centric tasks.",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
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
    description="Multilingual image embedding quality across 39 languages, spanning image classification (zero-shot and linear probing), clustering, retrieval, compositionality evaluation, document understanding, visual STS, and CV-centric tasks. Extends MIEB(eng) with multilingual retrieval datasets and the multilingual portions of VisualSTS-b and VisualSTS-16.",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
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
    description="Multilingual image embedding quality across the same task types as MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining relative model rankings.",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
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
    description="Image-only embedding quality across retrieval, classification, clustering, and visual STS, excluding tasks that require a text encoder.",
    reference="https://arxiv.org/abs/2504.10471",
    citation=r"""
@inproceedings{xiao2025mieb,
  author = {Xiao, Chenghao and Chung, Isaac and Kerboua, Imene and Stirling, Jamie and Zhang, Xin and Kardos, M\'arton and Solomatin, Roman and Al Moubayed, Noura and Enevoldsen, Kenneth and Muennighoff, Niklas},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month = {October},
  pages = {22187-22198},
  title = {MIEB: Massive Image Embedding Benchmark},
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
    description="Zero-shot retrieval quality in Polish across the BEIR task suite, covering biomedical, financial, legal, and general knowledge domains.",
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
    description="Text embedding quality in the built environment domain across clustering, retrieval, and reranking, spanning architecture, engineering, construction, and operations management.",
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
    description="Russian text embedding quality across paraphrase identification, sentiment analysis, toxicity classification, intent classification, natural language inference, and semantic similarity.",
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
    description="Visual document retrieval across diverse document types and domains, matching natural language queries to document page images.",
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
    description="Visual document retrieval across ESG reports, economics reports, biomedical lectures, and related enterprise document types, matching natural language queries to document page images.",
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
    description="Visual document retrieval across multi-modal enterprise documents spanning finance, industrial, computer science, pharmaceutical, and other professional domains. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml).",
    reference="https://arxiv.org/abs/2601.08620",
    citation=r"""
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
)

VIDORE_V3_1 = VidoreBenchmark(
    name="ViDoRe(v3.1)",
    display_name="ViDoRe v3.1",
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
            "Vidore3FinanceEnRetrieval.v2",
            "Vidore3IndustrialRetrieval.v2",
            "Vidore3ComputerScienceRetrieval.v2",
            "Vidore3PharmaceuticalsRetrieval.v2",
            "Vidore3HrRetrieval.v2",
            "Vidore3FinanceFrRetrieval.v2",
            "Vidore3PhysicsRetrieval.v2",
            "Vidore3EnergyRetrieval.v2",
            "Vidore3TelecomRetrieval.v2",
            "Vidore3NuclearRetrieval.v2",
        ]
    ),
    description="Visual document retrieval across multi-modal enterprise documents spanning finance, industrial, computer science, pharmaceutical, and other professional domains. Includes both open and closed datasets; to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml). v3.1 adds markdown derived from OCR to support text-only and joint image-text baselines.",
    reference="https://arxiv.org/abs/2601.08620",
    citation=r"""
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
)

VISUAL_DOCUMENT_RETRIEVAL = Benchmark(
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
    description="Visual document retrieval across diverse document types and domains, combining the ViDoRe v1 and v2 task sets.",
    reference="https://arxiv.org/abs/2407.01449",
    citation=r"""
@article{mace2025vidorev2,
  author = {Macé, Quentin and Loison António and Faysse, Manuel},
  journal = {arXiv preprint arXiv:2505.17166},
  title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
  year = {2025},
}
""",
    aggregations=(BenchmarkAggregation.MEAN_TASK,),
    show_zero_shot=False,
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
    description="Reasoning-driven medical retrieval quality across biology, bioinformatics, medical sciences, clinical, and treatment scenarios, requiring models to perform multi-step reasoning over medical literature.",
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
    description="Vietnamese text embedding quality across retrieval, classification, pair classification, clustering, reranking, and semantic similarity.",
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

MTEB_THA = Benchmark(
    name="MTEB(tha, v1)",
    aliases=["MTEB(tha)"],
    display_name="Thai",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/th.svg",
    tasks=get_tasks(
        languages=["tha"],
        tasks=[
            # Classification (4) — native Thai or purpose-built; no machine-translated tasks
            "MTOPDomainClassification",
            "MTOPIntentClassification",
            "SIB200Classification",
            "WisesightSentimentClassification.v2",
            # Clustering (1)
            "SIB200ClusteringS2S",
            # PairClassification (1)
            "XNLI",
            # Reranking (2)
            "MIRACLReranking",
            "MultiLongDocReranking",
            # Retrieval (7) — human-judged or human-translated
            "BelebeleRetrieval",
            "MIRACLRetrievalHardNegatives.v2",
            "MKQARetrieval",
            "MrTidyRetrieval",
            "MultiLongDocRetrieval",
            "WebFAQRetrieval",
            "XQuADRetrieval",
        ],
    ),
    description="Thai text embedding quality across classification, clustering, pair classification, reranking, and retrieval. Tasks are native Thai or high-quality human translations; machine-translated and cross-lingual tasks are excluded.",
    reference=None,
    citation=None,
    contacts=["anusoft"],
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
    description="Visual document retrieval across multilingual, domain-diverse, and layout-rich document types, spanning medical, legal, financial, technical, and other domains across multiple languages.",
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
    description="Text embedding performance benchmarked against human annotator scores across classification, clustering, reranking, and semantic similarity tasks, capturing where models exceed or fall short of human-level judgment.",
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
    description="Japanese text embedding quality across clustering, classification, semantic similarity, retrieval, and reranking. v2 extends the benchmark to 28 datasets for more comprehensive evaluation compared with MTEB(jpn, v1).",
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
    description="Japanese text embedding quality across clustering, classification, semantic similarity, retrieval, and reranking, with heavy datasets optimized via hard negative pooling to enable faster evaluation while maintaining rankings consistent with JMTEB.",
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

KOVIDORE_V2 = Benchmark(
    name="KoViDoRe(v2)",
    display_name="KoViDoRe v2",
    tasks=get_tasks(
        tasks=[
            "KoVidore2CybersecurityRetrieval",
            "KoVidore2EconomicRetrieval",
            "KoVidore2EnergyRetrieval",
            "KoVidore2HrRetrieval",
        ]
    ),
    description="Korean visual document retrieval across enterprise document domains including cybersecurity, economics, energy, and HR.",
    reference="https://github.com/whybe-choi/kovidore-data-generator",
    citation=r"""
@misc{choi2026kovidorev2,
  author = {Yongbin Choi},
  note = {A benchmark for evaluating Korean vision document retrieval with multi-page reasoning queries in practical domains},
  title = {KoViDoRe v2: a comprehensive evaluation of vision document retrieval for enterprise use-cases},
  url = {https://github.com/whybe-choi/kovidore-data-generator},
  year = {2026},
}
""",
)

VISRAG_RETRIEVAL = Benchmark(
    name="VisRAG Retrieval(v1)",
    aliases=["VisRAG", "VisRAG Retrieval", "VisRag", "VisRAG(v1)"],
    display_name="VisRAG Retrieval",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                "VisRAGRetArxivQA",
                "VisRAGRetChartQA",
                "VisRAGRetInfoVQA",
                "VisRAGRetPlotQA",
                "VisRAGRetMPDocVQA",
                "VisRAGRetSlideVQA",
            ],
        )
    ),
    description="Visual retrieval augmented generation quality across document types including scientific papers, charts, infographics, plots, and multi-page documents.",
    reference="https://huggingface.co/collections/openbmb/visrag",
    citation=r"""
@misc{yu2025visragvisionbasedretrievalaugmentedgeneration,
  archiveprefix = {arXiv},
  author = {Shi Yu and Chaoyue Tang and Bokai Xu and Junbo Cui and Junhao Ran and Yukun Yan and Zhenghao Liu and Shuo Wang and Xu Han and Zhiyuan Liu and Maosong Sun},
  eprint = {2410.10594},
  primaryclass = {cs.IR},
  title = {VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents},
  url = {https://arxiv.org/abs/2410.10594},
  year = {2025},
}
""",
)

_MAEB_CITATION = """@misc{assadi2026maebmassiveaudioembedding,
  archiveprefix = {arXiv},
  author = {Adnan El Assadi and Isaac Chung and Chenghao Xiao and Roman Solomatin and Animesh Jha and Rahul Chand and Silky Singh and Kaitlyn Wang and Ali Sartaz Khan and Marc Moussa Nasser and Sufen Fong and Pengfei He and Alan Xiao and Ayush Sunil Munot and Aditya Shrivastava and Artem Gazizov and Niklas Muennighoff and Kenneth Enevoldsen},
  eprint = {2602.16008},
  primaryclass = {cs.SD},
  title = {MAEB: Massive Audio Embedding Benchmark},
  url = {https://arxiv.org/abs/2602.16008},
  year = {2026},
}"""


MAEB_AUDIO = Benchmark(
    name="MAEB(beta, audio-only)",
    aliases=["MAEB(audio-only)"],
    display_name="MAEB Audio-Only",
    icon="https://raw.githubusercontent.com/DennisSuitters/LibreICONS/master/svg/libre-gui-activity.svg",
    tasks=get_tasks(
        tasks=[
            # Any2AnyRetrieval (1)
            "JamAltArtistA2ARetrieval",
            # AudioClassification (11)
            "BeijingOpera",
            "BirdCLEF",
            "CREMA_D",
            "CommonLanguageAgeDetection",
            "GTZANGenre",
            "IEMOCAPGender",
            "MInDS14",
            "MridinghamTonic",
            "SIBFLEURS",
            "VoxCelebSA",
            "VoxPopuliLanguageID",
            # AudioClustering (3)
            "CREMA_DClustering",
            "VehicleSoundClustering",
            "VoxPopuliGenderClustering",
            # AudioPairClassification (3)
            "CREMADPairClassification",
            "NMSQAPairClassification",
            "VoxPopuliAccentPairClassification",
            # AudioReranking (1)
            "GTZANAudioReranking",
        ]
    ),
    description="Audio-only embedding quality across classification, clustering, pair classification, reranking, and retrieval tasks. Currently in beta pending peer review.",
    reference=None,
    citation=_MAEB_CITATION,
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)
MAEB = Benchmark(
    name="MAEB(beta)",
    aliases=["MAEB"],
    display_name="MAEB",
    icon="https://raw.githubusercontent.com/DennisSuitters/LibreICONS/master/svg/libre-gui-activity.svg",
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                # Any2AnyRetrieval (9)
                "ClothoT2ARetrieval",
                "CommonVoiceMini21T2ARetrieval",
                "FleursT2ARetrieval",
                "GigaSpeechT2ARetrieval",
                "JamAltArtistA2ARetrieval",
                "JamAltLyricA2TRetrieval",
                "MACST2ARetrieval",
                "SpokenSQuADT2ARetrieval",
                "UrbanSound8KT2ARetrieval",
                # AudioClassification (11)
                "BeijingOpera",
                "BirdCLEF",
                "CREMA_D",
                "CommonLanguageAgeDetection",
                "GTZANGenre",
                "IEMOCAPGender",
                "MInDS14",
                "MridinghamTonic",
                "SIBFLEURS",
                "VoxCelebSA",
                "VoxPopuliLanguageID",
                # AudioClustering (3)
                "CREMA_DClustering",
                "VehicleSoundClustering",
                "VoxPopuliGenderClustering",
                # AudioPairClassification (3)
                "CREMADPairClassification",
                "NMSQAPairClassification",
                "VoxPopuliAccentPairClassification",
                # AudioReranking (1)
                "GTZANAudioReranking",
                # AudioZeroshotClassification (2)
                "RavdessZeroshot",
                "SpeechCommandsZeroshotv0.02",
            ]
        )
        + (
            # AudioMultilabelClassification (1), curated only (noisy has uncurated labels)
            get_task("FSD2019Kaggle", hf_subsets=["curated"]),
        )
    ),
    description="Audio embedding quality across both audio-only and audio-text cross-modal tasks, spanning retrieval, classification, clustering, multilabel classification, pair classification, reranking, and zero-shot classification. Currently in beta pending peer review.",
    reference=None,
    citation=_MAEB_CITATION,
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)


MVEB = Benchmark(
    name="MVEB(beta)",
    aliases=["MVEB"],
    display_name="MVEB",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-video.svg",
    tasks=get_tasks(
        tasks=[
            # Any2AnyRetrieval (10)
            "AVMemeExamAT2VRetrieval",
            "ActivityNetCaptionsT2VRetrieval",
            "AudioCapsAVVA2TRetrieval",
            "AudioCapsAVVT2ARetrieval",
            "MSVDT2VRetrieval",
            "VALOR32KT2VARetrieval",
            "VATEXV2ARetrieval",
            "VATEXVA2TRetrieval",
            "VGGSoundAVA2VRetrieval",
            "YouCook2T2VARetrieval",
            # VideoCentricQA (1)
            "EgoSchemaVideoCentricQA",
            # VideoClassification (6)
            "AVEDatasetClassification",
            "AVMemeAudioVideoClassification",
            "BreakfastClassification",
            "Kinetics700VA",
            "RAVDESSAVClassification",
            "UCF101VideoAudioClassification",
            # VideoClustering (2)
            "MELDEmotionAudioVideoClustering",
            "MusicAVQACLSAudioVideoClustering",
            # VideoPairClassification (2)
            "HumanAnimalCartoonVAPairClassification",
            "MusicAVQAVAPairClassification",
            # VideoZeroshotClassification (2)
            "HMDB51ZeroShot",
            "WorldSenseAudioVideoZeroShot",
        ]
    ),
    description="Audio-visual video embedding quality across retrieval, classification, clustering, pair classification, zero-shot classification, and video-centric QA, with tasks selected to maximize coverage of audio-video joint modality inputs.",
    reference=None,
    citation="",
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)


MVEB_TEXT_VIDEO = Benchmark(
    name="MVEB(text, video, beta)",
    aliases=["MVEB(text, video)"],
    display_name="MVEB Video-Text",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-video.svg",
    tasks=get_tasks(
        tasks=[
            # Any2AnyRetrieval (8)
            "AVMemeExamT2VRetrieval",
            "ActivityNetCaptionsT2VRetrieval",
            "AudioCapsAVT2VRetrieval",
            "DiDeMoV2TRetrieval",
            "MSVDV2TRetrieval",
            "Panda70MT2VRetrieval",
            "VALOR32KT2VRetrieval",
            "VATEXT2VRetrieval",
            # VideoCentricQA (1)
            "OmniVideoBenchVideoCentricQA",
            # VideoClassification (4)
            "AVMemeVideoClassification",
            "BreakfastClassification",
            "Kinetics700V",
            "VGGSoundV",
            # VideoClustering (1)
            "RAVDESSVideoClustering",
            # VideoPairClassification (1)
            "HumanAnimalCartoonVPairClassification",
            # VideoZeroshotClassification (4)
            "Kinetics400ZeroShot",
            "MELDVideoZeroShot",
            "UCF101VideoZeroShotClassification",
            "WorldSenseVideoZeroShot",
        ]
    ),
    description="Text and video embedding quality across retrieval, classification, clustering, pair classification, zero-shot classification, and video-centric QA, for models without an audio encoder.",
    reference=None,
    citation="",
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)


MVEB_VIDEO = Benchmark(
    name="MVEB(video, beta)",
    aliases=["MVEB(video)"],
    display_name="MVEB Video-Only",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-gui-video.svg",
    tasks=get_tasks(
        tasks=[
            # VideoClassification (6)
            "AVMemeVideoClassification",
            "BreakfastClassification",
            "HMDB51Classification",
            "Kinetics600V",
            "MELDVideoClassification",
            "WorldSenseVideoClassification",
            # VideoPairClassification (3)
            "HumanAnimalCartoonVPairClassification",
            "MusicAVQAVPairClassification",
            "RAVDESSAVVPairClassification",
        ]
    ),
    description="Video-only embedding quality across classification and pair classification, for encoders without a text component. Retrieval, QA, and zero-shot tasks are excluded as they require a text encoder.",
    reference=None,
    citation="",
    contacts=["AdnanElAssadi56", "isaac-chung", "KennethEnevoldsen", "Samoed"],
)

CoREB = Benchmark(
    name="CoREB(v1)",
    aliases=["CoREB"],
    display_name="CoREB",
    icon="https://github.com/DennisSuitters/LibreICONS/raw/2d2172d15e3c6ca03c018629d60050e4b99e5c55/svg-color/libre-tech-electronics.svg",
    tasks=get_tasks(
        tasks=[
            "CorebC2TRetrieval",
            "CorebC2CRetrieval",
            "CorebT2CRetrieval",
            "CorebC2TReranking",
            "CorebC2CReranking",
            "CorebT2CReranking",
        ],
    ),
    description="Code embedding and reranking quality across code-to-text, text-to-code, and code-to-code retrieval tasks, using counterfactually rewritten problems in five programming languages to limit training data contamination.",
    reference="https://arxiv.org/abs/2605.04615",
    citation=r"""
@article{xue2026coreb,
  author = {Xue, Siqiao and Liao, Zihan and Qin, Jin and Zhang, Ziyin and Mu, Yixiang and Zhou, Fan and Yu, Hang},
  journal = {arXiv preprint arXiv:2605.04615},
  title = {Beyond Retrieval: A Multitask Benchmark and Model for Code Search},
  url = {https://arxiv.org/abs/2605.04615},
  year = {2026},
}
""",
    contacts=["Geralt-Targaryen"],
)
