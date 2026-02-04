"""Test if the metadata of all tasks is filled and valid."""

import pytest

from mteb.abstasks import AbsTask
from mteb.get_tasks import get_tasks

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
_HISTORIC_DATASETS = [
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
    "AlloProfClusteringP2P",
    "AlloProfClusteringS2S",
    "MasakhaNEWSClusteringS2S",
    "SpanishNewsClusteringP2P",
    "SwednClustering",
    "CLSClusteringS2S",
    "CLSClusteringP2P",
    "ThuNewsClusteringS2S",
    "ThuNewsClusteringP2P",
    "GerDaLIR",
    "GerDaLIRSmall",
    "LegalQuAD",
    "AILACasedocs",
    "AILAStatutes",
    "CQADupstackRetrieval",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalSummarization",
    "NarrativeQARetrieval",
    "SCIDOCS",
    "SciFact",
    "TRECCOVID",
    "JaQuADRetrieval",
    "Ko-miracl",
    "Ko-StrategyQA",
    "MintakaRetrieval",
    "XMarket",
    "ArguAna-PL",
    "NFCorpus-PL",
    "NQ-PL",
    "Quora-PL",
    "SCIDOCS-PL",
    "SciFact-PL",
    "SpanishPassageRetrievalS2P",
    "SpanishPassageRetrievalS2S",
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "VideoRetrieval",
    "LeCaRDv2",
    "PawsX",
    "SICK-E-PL",
    "Ocnli",
    "Cmnli",
    "AskUbuntuDupQuestions",
    "SciDocsRR",
    "AlloprofReranking",
    "SyntecReranking",
    "T2Reranking",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "GermanSTSBenchmark",
    "SICK-R",
    "HUMESICK-R",
    "SICKFr",
    "KorSTS",
    "STSES",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STSB",
    "AFQMC",
    "QBQTC",
    "Quora-PLHardNegatives",
    "SynPerChatbotConvSAClassification",
    "CQADupstackRetrieval-Fa",
    "CQADupstackRetrieval-PL",
]


@pytest.mark.parametrize(
    "task", get_tasks(exclude_superseded=False, exclude_aggregate=False)
)
def test_all_metadata_is_filled_and_valid(task: AbsTask):
    # --- test metadata is filled and valid ---
    if task.metadata.name not in _HISTORIC_DATASETS:
        task.metadata._validate_metadata()
        assert task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is not filled"
        )
    else:
        assert not task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is stated as not filled (historic), but it is filled, please remove the dataset from the historic list."
        )

    # --- Check that no dataset trusts remote code ---
    assert task.metadata.dataset.get("trust_remote_code", False) is False, (
        f"Dataset {task.metadata.name} should not trust remote code"
    )

    # --- Test is descriptive stats are present for all datasets ---
    if task.is_aggregate:  # aggregate tasks do not have descriptive stats
        return

    assert task.metadata.descriptive_stats is not None, (
        f"Dataset {task.metadata.name} should have descriptive stats. You can add metadata to your task by running `YourTask().calculate_descriptive_statistics()`"
    )
    assert task.metadata.n_samples is not None
