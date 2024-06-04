from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Benchmark:
    name: str
    tasks: list[str]
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
    tasks=[
        "GeoreviewClassification",
        "GeoreviewClusteringP2P",
        "HeadlineClassification",
        "InappropriatenessClassification",
        "KinopoiskClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "RiaNewsRetrieval",
        "RuBQRetrieval",
        "RuReviewsClassification",
        "RuSciBenchGRNTIClassification",
        "RuSciBenchGRNTIClusteringP2P",
        "RuSciBenchOECDClassification",
        "RuSciBenchOECDClusteringP2P",
        "RuSTSBenchmarkSTS",
        "STS22",
        "TERRa",
    ],
    description="Main Russian benchmarks from MTEB",
    reference="https://aclanthology.org/2023.eacl-main.148/",
    citation=None,
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
    citation=None,  # TODO: add citation
)
