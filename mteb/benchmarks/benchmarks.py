from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import AnyUrl, BeforeValidator, TypeAdapter

from mteb.abstasks.AbsTask import AbsTask
from mteb.load_results.benchmark_results import BenchmarkResults
from mteb.load_results.load_results import load_results
from mteb.overview import MTEBTasks, get_task, get_tasks

http_url_adapter = TypeAdapter(AnyUrl)
UrlString = Annotated[
    str, BeforeValidator(lambda value: str(http_url_adapter.validate_python(value)))
]  # Allows the type to be a string, but ensures that the string is a URL


@dataclass
class Benchmark:
    """A benchmark object intended to run a certain benchmark within MTEB.

    Args:
        name: The name of the benchmark
        tasks: The tasks within the benchmark.
        description: A description of the benchmark, should include its intended goal and potentially a description of its construction
        reference: A link reference, to a source containing additional information typically to a paper, leaderboard or github.
        citation: A bibtex citation
        contacts: The people to contact in case of a problem in the benchmark, preferably a GitHub handle.

    Example:
        >>> Benchmark(
        ...     name="MTEB(custom)",
        ...     tasks=mteb.get_tasks(
        ...         tasks=["AmazonCounterfactualClassification", "AmazonPolarityClassification"],
        ...         languages=["eng"],
        ...     ),
        ...     description="A custom benchmark"
        ... )
    """

    name: str
    tasks: Sequence[AbsTask]
    description: str | None = None
    reference: UrlString | None = None
    citation: str | None = None
    contacts: list[str] | None = None

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]

    def load_results(
        self, base_results: None | BenchmarkResults = None
    ) -> BenchmarkResults:
        if not hasattr(self, "results_cache"):
            self.results_cache = {}
        if base_results in self.results_cache:
            return self.results_cache[base_results]
        if base_results is None:
            base_results = load_results()
        results = base_results.select_tasks(self.tasks)
        self.results_cache[base_results] = results
        return results


MTEB_EN = Benchmark(
    name="MTEB(eng)",
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

The original MTEB leaderboard is available under the [MTEB(eng, classic)](http://mteb-leaderboard-2-demo.hf.space/?benchmark_name=MTEB%28eng%2C+classic%29) tab.
    """,
    citation="",
    contacts=["KennethEnevoldsen", "Muennighoff"],
)

MTEB_ENG_CLASSIC = Benchmark(
    name="MTEB(eng, classic)",
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
            get_task("STS17", eval_splits=["test"], hf_subsets=["en-en"]),
            get_task("STS22", eval_splits=["test"], hf_subsets=["en"]),
        )
    ),
    description="""The original English benchmark by Muennighoff et al., (2023).
This page is an adaptation of the [old MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

> We recommend that you use [MTEB(eng)](http://mteb-leaderboard-2-demo.hf.space/?benchmark_name=MTEB%28eng%29) instead,
as many models have been tuned on MTEB(eng, classic) datasets, and MTEB(eng) might give a more accurate representation of models' generalization performance.
    """,
    citation="""@inproceedings{muennighoff-etal-2023-mteb,
    title = "{MTEB}: Massive Text Embedding Benchmark",
    author = "Muennighoff, Niklas  and
      Tazi, Nouamane  and
      Magne, Loic  and
      Reimers, Nils",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.148",
    doi = "10.18653/v1/2023.eacl-main.148",
    pages = "2014--2037",
}
""",
    contacts=["Muennighoff"],
)

MTEB_MAIN_RU = Benchmark(
    name="MTEB(rus)",
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
}
""",
)

MTEB_RETRIEVAL_WITH_INSTRUCTIONS = Benchmark(
    name="MTEB(Retrieval w/Instructions)",
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
    name="MTEB(law)",  # This benchmark is likely in the need of an update
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
    name="MTEB(Medical)",
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
    citation="""
    @article{winata2024miners,
    title={MINERS: Multilingual Language Models as Semantic Retrievers},
    author={Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
    journal={arXiv preprint arXiv:2406.07424},
    year={2024}
    }
    """,
)

SEB = Benchmark(
    name="MTEB(Scandinavian)",
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
            "DanFEVER",
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
    citation="""@misc{enevoldsen2024scandinavian,
      title={The Scandinavian Embedding Benchmarks: Comprehensive Assessment of Multilingual and Monolingual Text Embedding}, 
      author={Kenneth Enevoldsen and Márton Kardos and Niklas Muennighoff and Kristoffer Laigaard Nielbo},
      year={2024},
      eprint={2406.02396},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
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
    name="MTEB(fra)",
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
                "OpusparcusPC",
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
    name="MTEB(deu)",
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
    name="MTEB(kor)",
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
    name="MTEB(pol)",
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
                "STSBenchmarkMultilingualSTS",
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
    name="MTEB(code)",
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
    citation=None,
)


MTEB_multilingual = Benchmark(
    name="MTEB(Multilingual)",
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
    citation=None,
    contacts=["KennethEnevoldsen", "isaac-chung"],
)

MTEB_JPN = Benchmark(
    name="MTEB(jpn)",
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
    name="MTEB(Indic)",
    tasks=get_tasks(
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
            # STS
            "IndicCrosslingualSTS",
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
    ),
    description="A regional geopolitical text embedding benchmark targetting embedding performance on Indic languages.",
    reference=None,
    citation=None,
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
    name="MTEB(Europe)",
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
    citation=None,
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
    tasks=get_tasks(
        tasks=["BrightRetrieval"],
    ),
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
    name="MTEB(Chinese)",
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
        + get_tasks(tasks=["MultilingualSentiment"], eval_splits=["test"])
        + get_tasks(
            tasks=[
                "ATEC",
                "BQ",
                "STSB",
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
    name="FaMTEB(fas, beta)",
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
    citation="""
    @article{kasmaee2024chemteb,
    title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
    author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
    journal={arXiv preprint arXiv:2412.00532},
    year={2024}
}""",
)
