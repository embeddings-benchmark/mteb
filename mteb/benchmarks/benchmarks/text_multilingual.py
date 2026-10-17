"""Cross-lingual / multilingual aggregate text embedding benchmarks."""

from mteb.benchmarks.benchmark import (
    Benchmark,
    HUMEBenchmark,
)
from mteb.benchmarks.benchmarks._citations import MMTEB_CITATION
from mteb.get_tasks import MTEBTasks, get_tasks

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
