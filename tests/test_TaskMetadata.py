import pytest
from pydantic import ValidationError

from mteb.abstasks import AbsTask
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.overview import get_tasks

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
_HISTORIC_DATASETS = [
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "PolEmo2.0-OUT.v2",
    "PAC",
    "PAC.v2",
    "TNews",
    "TNews.v2",
    "IFlyTek",
    "IFlyTek.v2",
    "MultilingualSentiment",
    "MultilingualSentiment.v2",
    "JDReview",
    "JDReview.v2",
    "OnlineShopping",
    "Waimai",
    "Waimai.v2",
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
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
    "HUMESTSBenchmark",
    "HUMESICK-R",
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
    "ClimateFEVERHardNegatives",
    "DBPediaHardNegatives",
    "FEVERHardNegatives",
    "HotpotQAHardNegatives",
    "MSMARCOHardNegatives",
    "NQHardNegatives",
    "QuoraRetrievalHardNegatives",
    "TopiOCQAHardNegatives",
    "MIRACLRetrievalHardNegatives",
    "NeuCLIR2022RetrievalHardNegatives",
    "NeuCLIR2023RetrievalHardNegatives",
    "DBPedia-PLHardNegatives",
    "HotpotQA-PLHardNegatives",
    "MSMARCO-PLHardNegatives",
    "NQ-PLHardNegatives",
    "Quora-PLHardNegatives",
    "RiaNewsRetrievalHardNegatives",
    "SynPerChatbotConvSAClassification",
    "CQADupstackRetrieval-Fa",
    "IndicXnliPairClassification",
    "CQADupstackRetrieval-PL",
    "WikiClusteringP2P",
    "VGClustering",
    "VisualSTS17Eng",
    "VisualSTS17Multilingual",
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
        category="t2t",
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
            category="t2t",
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
            category="t2t",
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
            category="t2t",
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
            category="t2t",
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
            category="t2t",
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


@pytest.mark.parametrize(
    "task", get_tasks(exclude_superseded=True, exclude_aggregate=True)
)
def test_all_metadata_is_filled_and_valid(task: AbsTask):
    # --- test metadata is filled and valid ---
    if task.metadata.name not in _HISTORIC_DATASETS:
        task.metadata._validate_metadata()
        assert task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is not filled"
        )

    # --- Check that no dataset trusts remote code ---
    assert task.metadata.dataset.get("trust_remote_code", False) is False, (
        f"Dataset {task.metadata.name} should not trust remote code"
    )

    # --- Test is descriptive stats are present for all datasets ---
    if "image" in task.metadata.modalities:
        return

    # TODO add descriptive_stat for CodeRAGStackoverflowPosts. Required > 128GB of RAM
    if task.metadata.name in ["CodeRAGStackoverflowPosts"]:
        return

    assert task.metadata.descriptive_stats is not None, (
        f"Dataset {task.metadata.name} should have descriptive stats. You can add metadata to your task by running `YorTask().calculate_descriptive_statistics()`"
    )
    assert task.metadata.n_samples is not None
