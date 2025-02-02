from __future__ import annotations

import pytest
from pydantic import ValidationError

from mteb.abstasks import AbsTask, TaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.overview import get_tasks

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
_HISTORIC_DATASETS = [
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "AllegroReviews",
    "PAC",
    "TNews",
    "IFlyTek",
    "MultilingualSentiment",
    "JDReview",
    "OnlineShopping",
    "Waimai",
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BigPatentClustering",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
    "WikiCitiesClustering",
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "HALClusteringS2S",
    "MLSUMClusteringP2P",
    "MLSUMClusteringS2S",
    "MasakhaNEWSClusteringP2P",
    "MasakhaNEWSClusteringS2S",
    "EightTagsClustering",
    "RomaniBibleClustering",
    "SpanishNewsClusteringP2P",
    "SwednClustering",
    "CLSClusteringS2S",
    "CLSClusteringP2P",
    "ThuNewsClusteringS2S",
    "ThuNewsClusteringP2P",
    "TV2Nordretrieval",
    "TwitterHjerneRetrieval",
    "GerDaLIR",
    "GerDaLIRSmall",
    "GermanDPR",
    "GermanQuAD-Retrieval",
    "LegalQuAD",
    "AILACasedocs",
    "AILAStatutes",
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackRetrieval",
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
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HagridRetrieval",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalSummarization",
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "MSMARCO",
    "MSMARCOv2",
    "NarrativeQARetrieval",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "AlloprofRetrieval",
    "BSARDRetrieval",
    "SyntecRetrieval",
    "JaQuADRetrieval",
    "Ko-miracl",
    "Ko-StrategyQA",
    "MintakaRetrieval",
    "MIRACLRetrieval",
    "MultiLongDocRetrieval",
    "XMarket",
    "SNLRetrieval",
    "ArguAna-PL",
    "DBPedia-PL",
    "FiQA-PL",
    "HotpotQA-PL",
    "MSMARCO-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "TRECCOVID-PL",
    "SpanishPassageRetrievalS2P",
    "SpanishPassageRetrievalS2S",
    "SweFaqRetrieval",
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "VideoRetrieval",
    "LeCaRDv2",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "OpusparcusPC",
    "PawsX",
    "SICK-E-PL",
    "PpcPC",
    "CDSC-E",
    "PSC",
    "Ocnli",
    "Cmnli",
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
    "AlloprofReranking",
    "SyntecReranking",
    "T2Reranking",
    "MMarcoReranking",
    "CMedQAv1-reranking",
    "CMedQAv2-reranking",
    "GermanSTSBenchmark",
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "FinParaSTS",
    "SICKFr",
    "KLUE-STS",
    "KorSTS",
    "STS17",
    "STS22",
    "STSBenchmarkMultilingualSTS",
    "SICK-R-PL",
    "CDSC-R",
    "RonSTS",
    "STSES",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STSB",
    "AFQMC",
    "QBQTC",
    "SummEval",
    "SummEvalFr",
    "MalayalamNewsClassification",
    "TamilNewsClassification",
    "TenKGnadClusteringP2P.v2",
    "TenKGnadClusteringS2S.v2",
    "SynPerChatbotConvSAClassification",
    "CQADupstackRetrieval-Fa",
    "IndicXnliPairClassification",
]


def test_given_dataset_config_then_it_is_valid():
    my_task = TaskMetadata(
        name="MyTask",
        dataset={
            "path": "test/dataset",
            "revision": "1.0",
        },
        description="testing",
        reference=None,
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map",
        date=None,
        domains=None,
        license=None,
        task_subtypes=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="",
    )
    assert my_task.dataset["path"] == "test/dataset"
    assert my_task.dataset["revision"] == "1.0"


def test_given_missing_dataset_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(  # type: ignore
            name="MyTask",
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_given_missing_revision_path_then_it_throws():
    with pytest.raises(ValueError):
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_given_none_revision_path_then_it_logs_warning(caplog):
    with pytest.raises(ValidationError):
        TaskMetadata(
            name="MyTask",
            dataset={"path": "test/dataset", "revision": None},
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        )


def test_unfilled_metadata_is_not_filled():
    assert (
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference=None,
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=None,
            domains=None,
            license=None,
            task_subtypes=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation="",
        ).is_filled()
        is False
    )


def test_filled_metadata_is_filled():
    assert (
        TaskMetadata(
            name="MyTask",
            dataset={
                "path": "test/dataset",
                "revision": "1.0",
            },
            description="testing",
            reference="https://aclanthology.org/W19-6138/",
            type="Classification",
            category="s2s",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            main_score="map",
            date=("2021-01-01", "2021-12-31"),
            domains=["Non-fiction", "Written"],
            license="mit",
            task_subtypes=["Thematic clustering"],
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="found",
            bibtex_citation="Someone et al",
        ).is_filled()
        is True
    )


def test_all_metadata_is_filled_and_valid():
    all_tasks = get_tasks()

    unfilled_metadata = []
    for task in all_tasks:
        if (
            task.metadata.name not in _HISTORIC_DATASETS
            and task.metadata.name.replace("HardNegatives", "")
            not in _HISTORIC_DATASETS
        ):
            if not task.metadata.is_filled() and (
                not task.metadata.validate_metadata()
            ):
                unfilled_metadata.append(task.metadata.name)
    if unfilled_metadata:
        raise ValueError(
            f"The metadata of the following datasets is not filled: {unfilled_metadata}"
        )


def test_disallow_trust_remote_code_in_new_datasets():
    # DON'T ADD NEW DATASETS TO THIS LIST
    # THIS IS ONLY INTENDED FOR HISTORIC DATASETS
    exceptions = [
        "BornholmBitextMining",
        "BibleNLPBitextMining",
        "DiaBlaBitextMining",
        "FloresBitextMining",
        "IN22ConvBitextMining",
        "NTREXBitextMining",
        "IN22GenBitextMining",
        "IndicGenBenchFloresBitextMining",
        "IWSLT2017BitextMining",
        "SRNCorpusBitextMining",
        "VieMedEVBitextMining",
        "HotelReviewSentimentClassification",
        "TweetEmotionClassification",
        "DanishPoliticalCommentsClassification",
        "TenKGnadClassification",
        "ArxivClassification",
        "FinancialPhrasebankClassification",
        "FrenkEnClassification",
        "PatentClassification",
        "PoemSentimentClassification",
        "TweetTopicSingleClassification",
        "YahooAnswersTopicsClassification",
        "FilipinoHateSpeechClassification",
        "HebrewSentimentAnalysis",
        "HindiDiscourseClassification",
        "FrenkHrClassification",
        "Itacola",
        "JavaneseIMDBClassification",
        "WRIMEClassification",
        "KorHateClassification",
        "KorSarcasmClassification",
        "AfriSentiClassification",
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "NaijaSenti",
        "NordicLangClassification",
        "NusaX-senti",
        "SwissJudgementClassification",
        "MyanmarNews",
        "DutchBookReviewSentimentClassification",
        "NorwegianParliamentClassification",
        "PAC",
        "HateSpeechPortugueseClassification",
        "Moroco",
        "RomanianReviewsSentiment",
        "RomanianSentimentClassification",
        "GeoreviewClassification",
        "FrenkSlClassification",
        "DalajClassification",
        "SwedishSentimentClassification",
        "WisesightSentimentClassification",
        "UrduRomanSentimentClassification",
        "VieStudentFeedbackClassification",
        "IndicReviewsClusteringP2P",
        "MasakhaNEWSClusteringP2P",
        "MasakhaNEWSClusteringS2S",
        "MLSUMClusteringP2P.v2",
        "CodeSearchNetRetrieval",
        "DanFEVER",
        "GerDaLIR",
        "GermanDPR",
        "AlphaNLI",
        "ARCChallenge",
        "FaithDial",
        "HagridRetrieval",
        "HellaSwag",
        "PIQA",
        "Quail",
        "RARbCode",
        "RARbMath",
        "SIQA",
        "SpartQA",
        "TempReasonL1",
        "TempReasonL2Context",
        "TempReasonL2Fact",
        "TempReasonL2Pure",
        "TempReasonL3Context",
        "TempReasonL3Fact",
        "TempReasonL3Pure",
        "TopiOCQA",
        "WinoGrande",
        "AlloprofRetrieval",
        "BSARDRetrieval",
        "JaGovFaqsRetrieval",
        "JaQuADRetrieval",
        "NLPJournalAbsIntroRetrieval",
        "NLPJournalTitleAbsRetrieval",
        "NLPJournalTitleIntroRetrieval",
        "IndicQARetrieval",
        "MintakaRetrieval",
        "MIRACLRetrieval",
        "MLQARetrieval",
        "MultiLongDocRetrieval",
        "NeuCLIR2022Retrieval",
        "NeuCLIR2023Retrieval",
        "XMarket",
        "XPQARetrieval",
        "ArguAna-PL",
        "DBPedia-PL",
        "FiQA-PL",
        "HotpotQA-PL",
        "MSMARCO-PL",
        "NFCorpus-PL",
        "NQ-PL",
        "Quora-PL",
        "SCIDOCS-PL",
        "SciFact-PL",
        "TRECCOVID-PL",
        "SpanishPassageRetrievalS2P",
        "SpanishPassageRetrievalS2S",
        "SwednRetrieval",
        "SweFaqRetrieval",
        "KorHateSpeechMLClassification",
        "BrazilianToxicTweetsClassification",
        "CTKFactsNLI",
        "LegalBenchPC",
        "indonli",
        "OpusparcusPC",
        "PawsX",
        "XStance",
        "MIRACLReranking",
        "FinParaSTS",
        "JSICK",
        "JSTS",
        "RonSTS",
        "STSES",
        "AlloProfClusteringP2P.v2",
        "AlloProfClusteringS2S.v2",
        "LivedoorNewsClustering",
        "MewsC16JaClustering",
        "MLSUMClusteringS2S.v2",
        "SwednClusteringP2P",
        "SwednClusteringS2S",
        "IndicXnliPairClassification",
    ]

    assert (
        136 == len(exceptions)
    ), "The number of exceptions has changed. Please do not add new datasets to this list."

    exceptions = []

    for task in get_tasks():
        if task.metadata.dataset.get("trust_remote_code", False):
            assert (
                task.metadata.name not in exceptions
            ), f"Dataset {task.metadata.name} should not trust remote code"


@pytest.mark.parametrize("task", get_tasks())
def test_empty_descriptive_stat_in_new_datasets(task: AbsTask):
    if task.metadata.name.startswith("Mock") or isinstance(task, AbsTaskAggregate):
        return

    # TODO add descriptive_stat for CodeRAGStackoverflowPosts. Required > 128GB of RAM
    if task.metadata.name in ["CodeRAGStackoverflowPosts"]:
        return

    assert (
        task.metadata.descriptive_stats is not None
    ), f"Dataset {task.metadata.name} should have descriptive stats. You can add metadata to your task by running `YorTask().calculate_metadata_metrics()`"
    assert task.metadata.n_samples is not None


@pytest.mark.parametrize("task", get_tasks())
def test_eval_langs_correctly_specified(task: AbsTask):
    if task.is_multilingual:
        assert isinstance(
            task.metadata.eval_langs, dict
        ), f"{task.metadata.name} should have eval_langs as a dict"
    else:
        assert isinstance(
            task.metadata.eval_langs, list
        ), f"{task.metadata.name} should have eval_langs as a list"
