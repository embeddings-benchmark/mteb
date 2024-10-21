from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mteb.abstasks.AbsTask import AbsTask
from mteb.overview import get_tasks


@dataclass
class Benchmark:
    name: str
    tasks: Sequence[str] | Sequence[AbsTask]
    description: str | None = None
    reference: str | None = None
    citation: str | None = None

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index):
        return self.tasks[index]


MTEB_MAIN_EN = Benchmark(
    name="MTEB(eng)",
    tasks=[
        "AmazonCounterfactualClassification",
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
        "CQADupstackAndroidRetrieval",
        "CQADupstackEnglishRetrieval",
        "CQADupstackGamingRetrieval",
        "CQADupstackGisRetrieval",
        "CQADupstackMathematicaRetrieval",
        "CQADupstackPhysicsRetrieval",
        "CQADupstackProgrammersRetrieval",
        "CQADupstackStatsRetrieval",
        "CQADupstackTexRetrieval",
        "CQADupstackUnixRetrieval",
        "CQADupstackWebmastersRetrieval",
        "CQADupstackWordpressRetrieval",
        "ClimateFEVER",
        "DBPedia",
        "EmotionClassification",
        "FEVER",
        "FiQA2018",
        "HotpotQA",
        "ImdbClassification",
        "MSMARCO",
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
        "STS17",
        "STS22",
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
    description="Main English benchmarks from MTEB",
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
    description="Main Russian benchmarks from MTEB",
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
    tasks=[
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval",
        "Core17InstructionRetrieval",
    ],
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
    name="MTEB(law)",
    tasks=[
        "LegalSummarization",
        "LegalBenchConsumerContractsQA",
        "LegalBenchCorporateLobbying",
        "AILACasedocs",
        "AILAStatutes",
        "LeCaRDv2",
        "LegalQuAD",
        "GerDaLIRSmall",
    ],
    description="Legal benchmarks from MTEB",
    reference="https://aclanthology.org/2023.eacl-main.148/",
    citation=None,
)

MTEB_MINERS_BITEXT_MINING = Benchmark(
    name="MINERSBitextMining",
    tasks=[
        "BUCCBitextMining",
        "LinceMTBitextMining",
        "NollySentiBitextMining",
        "NusaXBitextMining",
        "NusaTranslationBitextMining",
        "PhincBitextMining",
        "TatoebaBitextMining",
    ],
    description="BitextMining benchmark from MINERS",
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
    tasks=[
        "BornholmBitextMining",
        "NorwegianCourtsBitextMining",
        "AngryTweetsClassification",
        "DanishPoliticalCommentsClassification",
        "DKHateClassification",
        "LccSentimentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "NordicLangClassification",
        "ScalaClassification",
        "NoRecClassification",
        "NorwegianParliamentClassification",
        "DalajClassification",
        "SwedishSentimentClassification",
        "SweRecClassification",
        "DanFEVER",
        "TV2Nordretrieval",
        "TwitterHjerneRetrieval",
        "NorQuadRetrieval",
        "SNLRetrieval",
        "SwednRetrieval",
        "SweFaqRetrieval",
        "WikiClusteringP2P.v2",
        "SNLHierarchicalClusteringP2P",
        "SNLHierarchicalClusteringS2S",
        "VGHierarchicalClusteringP2P",
        "VGHierarchicalClusteringS2S",
        "SwednClusteringP2P",
        "SwednClusteringS2S",
    ],
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
)

CoIR = Benchmark(
    name="CoIR",
    tasks=[
        "AppsRetrieval",
        "CosQA",
        "SyntheticText2SQL",
        "COIRCodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
        "CodeTransOceanDL",
        "CodeTransOceanContest",
        "StackOverflowQA",
        "CodeFeedbackMT",
        "CodeFeedbackST",
    ],
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

MTEB_FRA = Benchmark(
    name="MTEB(fra)",
    tasks=get_tasks(
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
            "SyntecReranking",
            "AlloprofReranking",
            # Retrieval
            "AlloprofRetrieval",
            "BSARDRetrieval",
            "SyntecRetrieval",
            "XPQARetrieval",
            "MintakaRetrieval",
            # STS
            "SummEvalFr",
            "STSBenchmarkMultilingualSTS",
            "STS22",
            "SICKFr",
        ],
    ),
    description="Main French benchmarks from MTEB",
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
)


MTEB_DEU = Benchmark(
    name="MTEB(deu)",
    tasks=get_tasks(
        languages=["deu"],
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
    description="Main German benchmarks from MTEB",
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
    description="Main Korean benchmarks from MTEB",
    reference=None,
    citation=None,
)


MTEB_pol = Benchmark(
    name="MTEB(pol)",
    tasks=get_tasks(
        languages=["pol"],
        tasks=[
            # Classification
            "CBD",
            "PolEmo2.0-IN",
            "PolEmo2.0-OUT",
            "AllegroReviews",
            "PAC",
            "MassiveIntentClassification",
            "MassiveScenarioClassification",
            # Clustering
            "EightTagsClustering",
            "PlscClusteringS2S",
            "PlscClusteringP2P",
            # Pair Classification
            "SICK-E-PL",
            "PpcPC",
            "CDSC-E",
            "PSC",
            # STS
            "SICK-R-PL",
            "CDSC-R",
            "STS22",
            "STSBenchmarkMultilingualSTS",
        ],
    ),
    description="Main Polish benchmarks from MTEB",
    reference="https://arxiv.org/abs/2405.10138",
    citation="""@article{poswiata2024plmteb,
    title={PL-MTEB: Polish Massive Text Embedding Benchmark},
    author={Rafał Poświata and Sławomir Dadas and Michał Perełkiewicz},
    journal={arXiv preprint arXiv:2405.10138},
    year={2024}
}""",
)
