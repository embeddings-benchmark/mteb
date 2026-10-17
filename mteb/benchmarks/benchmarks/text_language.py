"""Language- and region-specific text embedding benchmarks (the MTEB(<lang>) family and regional ports)."""

from mteb.benchmarks.benchmark import Benchmark
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
    display_name="English",
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
@misc{ciancone2024mteb,
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
    aliases=[],
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
    superseded_by=["MTEB(kor, v2)"],
)

MTEB_KOR_V2 = Benchmark(
    name="MTEB(kor, v2)",
    aliases=["MTEB(kor)"],
    display_name="Korean",
    icon="https://github.com/lipis/flag-icons/raw/260c91531be024944c6514130c5defb2ebb02b7d/flags/4x3/kr.svg",
    tasks=get_tasks(
        languages=["kor"],
        tasks=[
            # Classification
            "KLUE-TC",
            # Clustering
            "SIB200ClusteringS2S",
            "KlueMrcDomainClustering",
            "KlueYnatMrcCategoryClustering",
            # Pair Classification (NLI)
            "KLUE-NLI",
            "KorNLI",
            "PawsXPairClassification",
            # Reranking
            "MIRACLReranking",
            # Retrieval
            "MIRACLRetrieval",
            "Ko-StrategyQA",
            "LawIRKo",
            "SQuADKorV1Retrieval",
            "AutoRAGRetrieval",
            "PublicHealthQA",
            "BelebeleRetrieval",
            "MultiLongDocRetrieval",
            "MrTidyRetrieval",
            # STS
            "KLUE-STS",
            "KorSTS",
            "STS17",
        ],
    ),
    description=(
        "Korean text embedding quality across classification, clustering, pair "
        "classification (NLI), reranking, retrieval, and semantic similarity. "
        "Expanded successor to MTEB(kor, v1): grows the suite from 6 to 20 tasks, "
        "adding clustering (SIB200, KLUE-MRC domain/category), pair classification/NLI "
        "(KLUE-NLI, KorNLI, PawsX), additional retrieval (LawIRKo, SQuAD-Ko, AutoRAG, "
        "PublicHealthQA, Belebele, MultiLongDoc, MrTidy) and STS17, for broader and more "
        "robust coverage of Korean embedding capabilities."
    ),
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

MTEB_SLK = Benchmark(
    name="MTEB(slk, v1)",
    aliases=["MTEB(slk)"],
    display_name="Slovak",
    icon="https://github.com/lipis/flag-icons/raw/59d15a43b2e643fc94e519904d148c616dec20e7/flags/4x3/sk.svg",
    tasks=MTEBTasks(
        get_tasks(
            languages=["slk"],
            tasks=[
                # Retrieval
                "BelebeleRetrieval",
                "SlovakSumRetrieval",
                "SMESumRetrieval",
                "SKQuadRetrieval",
                "WebFAQRetrieval",
                # STS
                "SlovakSTS",
                "SlovakSumSTS",
                # Pair Classification
                "SlovakNLI",
                "SlovakRTE",
                "DemagogSKNLI",
                # Classification
                "SlovakHateSpeechClassification.v2",
                "SlovakMovieReviewSentimentClassification.v2",
                "SIB200Classification",
                "MultilingualSentimentClassification",
                "SlovakParlaSentClassification",
                "MultiEupSlovakPartyClassification",
                "MultiEupSlovakGenderClassification",
                # Reranking
                "SkQuadReranking",
                "SlovakPharmacyDrMaxReranking",
                "SlovakPharmacyMojaLekarenReranking",
                # Clustering
                "SIB200ClusteringS2S",
                "PravdaSKTagClustering",
                "PravdaSKURLClustering",
                "SlovakSumURLClustering",
                "SMESumCategoryClustering",
                # BitextMining
                "OpusSlovakEnglishBitextMining",
                "Tatoeba",
            ],
        )
        + (
            get_task(
                "FloresBitextMining",
                hf_subsets=["eng_Latn-slk_Latn", "ces_Latn-slk_Latn"],
            ),
        )
        + (
            get_task(
                "NTREXBitextMining",
                hf_subsets=["eng_Latn-slk_Latn", "ces_Latn-slk_Latn"],
            ),
        )
        + (get_task("WebFAQBitextMiningQuestions", hf_subsets=["eng-slk", "ces-slk"]),)
        + (get_task("WebFAQBitextMiningQAs", hf_subsets=["eng-slk", "ces-slk"]),)
    ),
    description="""Slovak Massive Text Embedding Benchmark (SkMTEB), the first comprehensive benchmark for text embeddings in Slovak, a low-resource West Slavic language. SkMTEB consists of 31 tasks across 7 task types, with tasks adapted from both existing and novel resources.""",
    reference="https://arxiv.org/abs/2606.13647",
    citation=r"""
@misc{suppa2026skmteb,
  archiveprefix = {arXiv},
  author = {Marek Šuppa and Andrej Ridzik and Daniel Hládek and Natália Kňažeková and Viktória Ondrejová},
  eprint = {2606.13647},
  primaryclass = {cs.CL},
  title = {SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation},
  url = {https://arxiv.org/abs/2606.13647},
  year = {2026},
}
""",
    contacts=["andrejridzik", "hladek", "mrshu"],
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

MTEB_JPN = Benchmark(
    name="MTEB(jpn, v1)",
    aliases=["MTEB(jpn)"],
    display_name="Japanese",
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
@misc{xiao2024cpackpackedresourcesgeneral,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packed Resources For General Chinese Embeddings},
  url = {https://arxiv.org/abs/2309.07597},
  year = {2024},
}
""",
)

FA_MTEB = Benchmark(
    name="MTEB(fas, v1)",
    aliases=["FaMTEB(fas, beta)"],
    display_name="Persian",
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
    superseded_by=["MTEB(fas, v2)"],
)

FA_MTEB_2 = Benchmark(
    name="MTEB(fas, v2)",
    display_name="Persian",
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
    citation=None,
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
@inproceedings{li-etal-2026-jmteb,
  address = {Palma de Mallorca, Spain},
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  doi = {10.63317/5ouzpv2f2f6k},
  month = may,
  pages = {7423--7434},
  publisher = {ELRA Language Resource Association},
  title = {{JMTEB} and {JMTEB}-lite: {J}apanese Massive Text Embedding Benchmark and Its Lightweight Version},
  url = {https://aclanthology.org/2026.lrec-1.588/},
  year = {2026},
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
@inproceedings{li-etal-2026-jmteb,
  address = {Palma de Mallorca, Spain},
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  doi = {10.63317/5ouzpv2f2f6k},
  month = may,
  pages = {7423--7434},
  publisher = {ELRA Language Resource Association},
  title = {{JMTEB} and {JMTEB}-lite: {J}apanese Massive Text Embedding Benchmark and Its Lightweight Version},
  url = {https://aclanthology.org/2026.lrec-1.588/},
  year = {2026},
}
""",
    contacts=["lsz05"],
)
