from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import AnyUrl, BeforeValidator, TypeAdapter

from mteb.benchmarks.benchmark import Benchmark
from mteb.overview import MTEBTasks, get_task, get_tasks

if TYPE_CHECKING:
    pass

http_url_adapter = TypeAdapter(AnyUrl)
UrlString = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL


MMTEB_CITATION = """@article{enevoldsen2025mmtebmassivemultilingualtext,
    title={MMTEB: Massive Multilingual Text Embedding Benchmark}, 
    author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
    publisher = {arXiv},
    journal={arXiv preprint arXiv:2502.13595},
    year={2025},
    url={https://arxiv.org/abs/2502.13595}, 
    doi = {10.48550/arXiv.2502.13595},
}"""

MTEB_EN = Benchmark(
    name="MTEB(eng, v2)",
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
    citation="""@article{muennighoff2022mteb,
    author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
    title = {MTEB: Massive Text Embedding Benchmark},
    publisher = {arXiv},
    journal={arXiv preprint arXiv:2210.07316},
    year = {2022}
    url = {https://arxiv.org/abs/2210.07316},
    doi = {10.48550/ARXIV.2210.07316},
}""",
    contacts=["Muennighoff"],
)

MTEB_MAIN_RU = Benchmark(
    name="MTEB(rus, v1)",
    tasks=get_tasks(
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
            "RuSTSBenchmarkSTS",
            "STS22",
        ],
    ),
    description="A Russian version of the Massive Text Embedding Benchmark with a number of novel Russian tasks in all task categories of the original MTEB.",
    reference="https://aclanthology.org/2023.eacl-main.148/",
    citation="""@misc{snegirev2024russianfocusedembeddersexplorationrumteb,
    title={The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design},
    author={Artem Snegirev and Maria Tikhonova and Anna Maksimova and Alena Fenogenova and Alexander Abramov},
    year={2024},
    eprint={2408.12503},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2408.12503},
}""",
)

MTEB_RETRIEVAL_WITH_INSTRUCTIONS = Benchmark(
    name="FollowIR",
    tasks=get_tasks(
        tasks=[
            "Robust04InstructionRetrieval",
            "News21InstructionRetrieval",
            "Core17InstructionRetrieval",
        ]
    ),
    description="Retrieval w/Instructions is the task of finding relevant documents for a query that has detailed instructions.",
    reference="https://arxiv.org/abs/2403.15246",
    citation="""@misc{weller2024followir,
    title={FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
    author={Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
    year={2024},
    eprint={2403.15246},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}""",
)

MTEB_RETRIEVAL_LAW = Benchmark(
    name="MTEB(Law, v1)",  # This benchmark is likely in the need of an update
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
    reference="",
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
    citation="""@article{winata2024miners,
    title={MINERS: Multilingual Language Models as Semantic Retrievers},
    author={Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
    journal={arXiv preprint arXiv:2406.07424},
    year={2024}
}""",
)

SEB = Benchmark(
    name="MTEB(Scandinavian, v1)",
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
    citation="""@inproceedings{enevoldsen2024scandinavian,
    title={The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding},
    author={Enevoldsen, Kenneth and Kardos, M{\'a}rton and Muennighoff, Niklas and Nielbo, Kristoffer},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
    url={https://nips.cc/virtual/2024/poster/97869}
}""",
    contacts=["KennethEnevoldsen", "x-tabdeveloping", "Samoed"],
)

CoIR = Benchmark(
    name="CoIR",
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
    citation="""@misc{li2024coircomprehensivebenchmarkcode,
    title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
    author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
    year={2024},
    eprint={2407.02883},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
    url={https://arxiv.org/abs/2407.02883},
}""",
)

RAR_b = Benchmark(
    name="RAR-b",
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
    citation="""@article{xiao2024rar,
    title={RAR-b: Reasoning as Retrieval Benchmark},
    author={Xiao, Chenghao and Hudson, G Thomas and Al Moubayed, Noura},
    journal={arXiv preprint arXiv:2404.06347},
    year={2024}
}""",
    contacts=["gowitheflow-1998"],
)

MTEB_FRA = Benchmark(
    name="MTEB(fra, v1)",
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
    citation="""@misc{ciancone2024mtebfrenchresourcesfrenchsentence,
    title={MTEB-French: Resources for French Sentence Embedding Evaluation and Analysis},
    author={Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
    year={2024},
    eprint={2405.20468},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2405.20468},
}""",
    contacts=["imenelydiaker"],
)

MTEB_DEU = Benchmark(
    name="MTEB(deu, v1)",
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
    citation="""@misc{wehrli2024germantextembeddingclustering,
    title={German Text Embedding Clustering Benchmark},
    author={Silvan Wehrli and Bert Arnrich and Christopher Irrgang},
    year={2024},
    eprint={2401.02709},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2401.02709},
}""",
    contacts=["slvnwhrl"],
)

MTEB_KOR = Benchmark(
    name="MTEB(kor, v1)",
    tasks=get_tasks(
        languages=["kor"],
        tasks=[  # @KennethEnevoldsen: We could probably expand this to a more solid benchamrk, but for now I have left it as is.
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
    citation="""@article{poswiata2024plmteb,
    title={PL-MTEB: Polish Massive Text Embedding Benchmark},
    author={Rafał Poświata and Sławomir Dadas and Michał Perełkiewicz},
    journal={arXiv preprint arXiv:2405.10138},
    year={2024}
}""",
    contacts=["rafalposwiata"],
)

MTEB_code = Benchmark(
    name="MTEB(Code, v1)",
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
        languages=[
            "c",
            "c++",
            "go",
            "java",
            "javascript",
            "php",
            "python",
            "ruby",
            "rust",
            "scala",
            "shell",
            "swift",
            "typescript",
        ],
    ),
    description="A massive code embedding benchmark covering retrieval tasks in a miriad of popular programming languages.",
    reference=None,
    citation=MMTEB_CITATION,
)

MTEB_multilingual = Benchmark(
    name="MTEB(Multilingual, v1)",
    tasks=get_tasks(
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
            "SNLHierarchicalClusteringP2P",
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
    ),
    description="A large-scale multilingual expansion of MTEB, driven mainly by highly-curated community contributions covering 250+ languages.",
    reference=None,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)

MTEB_JPN = Benchmark(
    name="MTEB(jpn, v1)",
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
    tasks=MTEBTasks(
        get_tasks(
            tasks=[
                # Bitext
                "IN22ConvBitextMining",
                "IN22GenBitextMining",
                "IndicGenBenchFloresBitextMining",
                "LinceMTBitextMining",
                # clustering
                "SIB200ClusteringS2S",
                # classification
                "BengaliSentimentAnalysis",
                "GujaratiNewsClassification",
                "HindiDiscourseClassification",
                "SentimentAnalysisHindi",
                "MalayalamNewsClassification",
                "IndicLangClassification",
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
    description="A regional geopolitical text embedding benchmark targetting embedding performance on Indic languages.",
    reference=None,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)


eu_languages = [
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
            "NordicLangClassification",
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
    description="A regional geopolitical text embedding benchmark targetting embedding performance on European languages.",
    reference=None,
    citation=MMTEB_CITATION,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)

LONG_EMBED = Benchmark(
    name="LongEmbed",
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
    citation="""@article{zhu2024longembed,
    title={LongEmbed: Extending Embedding Models for Long Context Retrieval},
    author={Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
    journal={arXiv preprint arXiv:2404.12096},
    year={2024}
}""",
)

BRIGHT = Benchmark(
    name="BRIGHT",
    tasks=get_tasks(tasks=["BrightRetrieval"], eval_splits=["standard"]),
    description="""BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.
    BRIGHT is the first text retrieval
    benchmark that requires intensive reasoning to retrieve relevant documents with
    a dataset consisting of 1,384 real-world queries spanning diverse domains, such as
    economics, psychology, mathematics, and coding. These queries are drawn from
    naturally occurring and carefully curated human data.
    """,
    reference="https://brightbenchmark.github.io/",
    citation="""@article{su2024bright,
    title={Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
    author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
    journal={arXiv preprint arXiv:2407.12883},
    year={2024}
}""",
)

BRIGHT_LONG = Benchmark(
    name="BRIGHT (long)",
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
    citation="""@article{su2024bright,
    title={Bright: A realistic and challenging benchmark for reasoning-intensive retrieval},
    author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and others},
    journal={arXiv preprint arXiv:2407.12883},
    year={2024}
}""",
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
    citation="""@misc{wang2024coderagbenchretrievalaugmentcode,
    title={CodeRAG-Bench: Can Retrieval Augment Code Generation?},
    author={Zora Zhiruo Wang and Akari Asai and Xinyan Velocity Yu and Frank F. Xu and Yiqing Xie and Graham Neubig and Daniel Fried},
    year={2024},
    eprint={2406.14497},
    archivePrefix={arXiv},
    primaryClass={cs.SE},
    url={https://arxiv.org/abs/2406.14497},
}""",
    display_on_leaderboard=False,
)

BEIR = Benchmark(
    name="BEIR",
    tasks=get_tasks(
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
    + get_tasks(tasks=["MSMARCO"], languages=["eng"], eval_splits=["dev"]),
    description="BEIR is a heterogeneous benchmark containing diverse IR tasks. It also provides a common and easy framework for evaluation of your NLP-based retrieval models within the benchmark.",
    reference="https://arxiv.org/abs/2104.08663",
    citation="""@article{thakur2021beir,
    title={Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models},
    author={Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
    journal={arXiv preprint arXiv:2104.08663},
    year={2021}
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
    citation="""@misc{c-pack,
    title={C-Pack: Packaged Resources To Advance General Chinese Embedding},
    author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
    year={2023},
    eprint={2309.07597},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}""",
)

FA_MTEB = Benchmark(
    name="MTEB(fas, beta)",
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
            "PersianTextTone",
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
    description="Main Persian (Farsi) benchmarks from MTEB",
    reference=None,
    citation=None,
    contacts=["mehran-sarmadi", "ERfun", "morteza20"],
)

CHEMTEB = Benchmark(
    name="ChemTEB",
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
    citation="""@article{kasmaee2024chemteb,
    title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \\& Efficiency on a Specific Domain},
    author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
    journal={arXiv preprint arXiv:2412.00532},
    year={2024}
}""",
)

BEIR_NL = Benchmark(
    name="BEIR-NL",
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
    citation="""@misc{banar2024beirnlzeroshotinformationretrieval,
    title={BEIR-NL: Zero-shot Information Retrieval Benchmark for the Dutch Language},
    author={Nikolay Banar and Ehsan Lotfi and Walter Daelemans},
    year={2024},
    eprint={2412.08329},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2412.08329},
}""",
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

MIEB_ENG = Benchmark(
    name="MIEB(eng)",
    tasks=get_tasks(
        tasks=MIEB_common_tasks
        + [
            "VisualSTS17Eng",
            "VisualSTS-b-Eng",
        ],
    ),
    description="""MIEB(eng) is a comprehensive image embeddings benchmark, spanning 8 task types, covering 125 tasks.
    In addition to image classification (zero shot and linear probing), clustering, retrieval, MIEB includes tasks in compositionality evaluation,
    document undestanding, visual STS, and CV-centric tasks.""",
    reference="",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation="",
)

MIEB_MULTILINGUAL = Benchmark(
    name="MIEB(Multilingual)",
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
    document undestanding, visual STS, and CV-centric tasks. This benchmark consists of MIEB(eng) + 3 multilingual retrieval
    datasets + the multilingual parts of VisualSTS-b and VisualSTS-16.""",
    reference="",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation="",
)

MIEB_LITE = Benchmark(
    name="MIEB(lite)",
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
            "STS17MultilingualVisualSTS",
            "STSBenchmarkMultilingualVisualSTS",
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
    reference="",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation="",
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
    description='"Built-Bench" is an ongoing effort aimed at evaluating text embedding models in the context of built asset management, spanning over various dicsiplines such as architeture, engineering, constrcution, and operations management of the built environment.',
    reference="https://arxiv.org/abs/2411.12056",
    citation="""@article{shahinmoghadam2024benchmarking,
    title={Benchmarking pre-trained text embedding models in aligning built asset information},
    author={Shahinmoghadam, Mehrzad and Motamedi, Ali},
    journal={arXiv preprint arXiv:2411.12056},
    year={2024}
}""",
    contacts=["mehrzadshm"],
)
