from __future__ import annotations

import pytest

import mteb
from mteb import AbsTask


# @pytest.mark.parametrize(
#     "model_name",
#     [
#         "jinaai/jina-embeddings-v3",
#     ],
# )
# def test_get_model(model_name):
#     mteb.get_model(model_name)


# @pytest.mark.parametrize(
#     "task_name",
#     [
#         # "TbilisiCityHallBitextMining",
#         # "TwentyNewsgroupsClustering.v2",
#         # "StackExchangeClusteringP2P.v2",
#         # "RedditClusteringP2P.v2",
#         # "MedrxivClusteringS2S.v2",
#         # "MedrxivClusteringP2P.v2",
#         # "BornholmBitextMining",
#         # "IN22ConvBitextMining",
#         # "NusaTranslationBitextMining",
#         # "LanguageClassification",
#         # "ArXivHierarchicalClusteringP2P",
#         # "BiorxivClusteringS2S",
#         # "WikiClusteringP2P",
#         # "RuSciBenchGRNTIClusteringP2P",
#         # "Core17InstructionRetrieval",
#         # "MultiEURLEXMultilabelClassification",
#         # "CEDRClassification",
#         # "TwitterURLCorpus",
#         # "PawsXPairClassification",
#         # "XNLI",
#         # "AskUbuntuDupQuestions",
#         # "WikipediaRerankingMultilingual",
#         # "AppsRetrieval",
#         # "BelebeleRetrieval",
#         # "STS12",
#         # "STS17",
#         # "SummEval",
#         # "ESCIReranking",
#         # "Touche2020Retrieval.v3",
#         # "JaqketRetrieval",
#         # "SlovakHateSpeechClassification",
#         # "mFollowIRCrossLingual",
#         # "mFollowIR",
#         # "InstructIR",
#         # "NevIR",
#         # "CodeEditSearchRetrieval",
#         # "CodeFeedbackMT",
#         # "CodeFeedbackST",
#         # "CodeSearchNetCCRetrieval",
#         # "CodeSearchNetRetrieval",
#         # "CodeTransOceanContest",
#         # "CodeTransOceanDL",
#         # "COIRCodeSearchNetRetrieval",
#         # "CosQA",
#         # "StackOverflowQA",
#         # "SyntheticText2SQL",
#         # "Touche2020",
#         # # # bitext
#         # "PhincBitextMining",
#         # "NusaTranslationBitextMining",
#         # # "BibleNLPBitextMining",
#         # "NTREXBitextMining",
#         # "IN22ConvBitextMining",
#         # "IWSLT2017BitextMining",
#         # "BUCC.v2",
#         # "BornholmBitextMining",
#         # "NollySentiBitextMining",
#         # "TbilisiCityHallBitextMining",
#         # "IndicGenBenchFloresBitextMining",
#         # # "FloresBitextMining",
#         # "NorwegianCourtsBitextMining",
#         # "VieMedEVBitextMining",
#         # "IN22GenBitextMining",
#         # "CEDRClassification",
#     ],
# )
# def test_descriptive_task(task_name):
#     task = mteb.get_task(task_name)
#     task.calculate_metadata_metrics(overwrite_results=True)

# remote code Hellaswag, PIQA, Quail, SIQA, SpartQA, TempReasonL1, TempReasonL2Context, TempReasonL2Fact, TempReasonL2Pure, TempReasonL3Context, TempReasonL3Fact, TempReasonL3Pure,  WinoGrande, RarbCode, rarbmath, SIQA, Spartqa, NeuCLIR2022Retrieval, NeuCLIR2023Retrieval
# Failed IndicQARetrieval, MIRACLRetrievalHardNegatives, NeuCLIR2023RetrievalHardNegatives, XMarket, SpanishPassageRetrievalS2P, T2Retrieval
@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["Retrieval"]))
def test_process_descripitve_stats_retrieval(task: AbsTask):
    print(task.metadata.name)
    if task.metadata.name in ["BrightRetrieval", "FEVER", "BelebeleRetrieval", "HotpotQA", "MSMARCO", "MSMARCOv2",
                              "MIRACLRetrieval", "MrTidyRetrieval", "TopiOCQA", "MultiLongDocRetrieval",
                              "NeuCLIR2022Retrieval", "NeuCLIR2023Retrieval"]:
        return
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["Reranking"]))
def test_process_descripitve_stats_reranking(task):
    print(task.metadata.name)
    if task.metadata.name in ["MIRACLReranking", "MindSmallReranking", "VoyageMMarcoReranking",
                              "WebLINXCandidatesReranking"]:
        return
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["InstructionReranking", "InstructionRetrieval"]))
def test_process_descripitve_stats_instruct(task):
    print(task.metadata.name)
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["Clustering"]))
def test_process_descripitve_stats_clustering(task):
    print(task.metadata.name)
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["Classification"]))
def test_process_descripitve_stats_classification(task):
    print(task.metadata.name)
    if task.metadata.name in ["SwissJudgementClassification"]:
        return
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["MultilabelClassification"]))
def test_process_descripitve_stats_multilabelclassification(task):
    print(task.metadata.name)
    if task.metadata.name in ["MultiEURLEXMultilabelClassification"]:
        return
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["PairClassification"]))
def test_process_descripitve_stats_pairclassification(task):
    print(task.metadata.name)
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["BitextMining"]))
def test_process_descripitve_stats_bitextmining(task):
    print(task.metadata.name)
    if task.metadata.name in [
        "BibleNLPBitextMining",
        "FloresBitextMining"
    ]:
        return
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["STS"]))
def test_process_descripitve_stats_sts(task):
    print(task.metadata.name)
    if task.metadata.name in ["MultiEURLEXMultilabelClassification"]:
        return
    task.calculate_metadata_metrics()


@pytest.mark.parametrize("task", mteb.get_tasks(task_types=["Summarization"]))
def test_process_descripitve_stats_summarization(task):
    print(task.metadata.name)
    if task.metadata.name in ["MultiEURLEXMultilabelClassification"]:
        return
    task.calculate_metadata_metrics()

# ["BrightRetrieval", "FEVER", "BelebeleRetrieval", "HotpotQA", "MSMARCO", "MSMARCOv2", "MIRACLRetrieval", "MrTidyRetrieval", "TopiOCQA", "MultiLongDocRetrieval", "NeuCLIR2022Retrieval", "NeuCLIR2023Retrieval"] + ["MultiEURLEXMultilabelClassification", "SwissJudgementClassification"] + ["BibleNLPBitextMining","FloresBitextMining"] + ["MIRACLReranking", "MindSmallReranking", "VoyageMMarcoReranking", "WebLINXCandidatesReranking"]
