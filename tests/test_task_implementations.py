from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest
import srsly

from mteb import MTEB
from mteb.abstasks import AbsTask
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


@dataclass(frozen=True)
class HFSpecificationError(Exception):
    name: str
    error: str


def check_hf_lang_configuration_is_valid(
    task: AbsTask,
) -> Sequence[HFSpecificationError]:
    errors: list[HFSpecificationError] = []
    try:
        task.load_data()
    except Exception as e:
        return [HFSpecificationError(task.metadata.name, str(e))]

    if isinstance(task, AbsTaskRetrieval):
        if not task.corpus:
            errors.append(
                HFSpecificationError(
                    task.metadata.name, "self.corpus is empty or not set"
                )
            )
        if not task.queries:
            errors.append(
                HFSpecificationError(
                    task.metadata.name, "self.queries is empty or not set"
                )
            )
        if not task.relevant_docs:
            errors.append(
                HFSpecificationError(
                    task.metadata.name, "self.relevant_docs is empty or not set"
                )
            )
    else:
        for split in task.metadata.eval_splits:
            try:
                if split not in task.dataset.keys():  # type: ignore
                    errors.append(
                        HFSpecificationError(
                            task.metadata.name, f"Split {split} not found"
                        )
                    )
            except Exception as e:
                errors.append(HFSpecificationError(task.metadata.name, str(e)))

    return errors


def check_task_hf_specification(
    task: AbsTask,
) -> Sequence[HFSpecificationError]:
    # Check if task has been already validated
    # Copy so you do not mutate the original metadata
    cache_key = copy.deepcopy(task.metadata.dataset)

    # Add the task name to the dataset metadata so it becomes part of cache invalidation
    # E.g. some tasks have the same dataset but different splits, so we want to check them independently
    cache_key["task_name"] = task.metadata.name
    cache_path = Path(__file__).parent / "checked_hf_specifications.jsonl"
    if cache_key in list(srsly.read_jsonl(cache_path)):
        return []

    logging.info(f"Checking task {task.metadata.name}")
    errors = check_hf_lang_configuration_is_valid(task=task)

    # Add to the cache
    srsly.write_jsonl(cache_path, [cache_key], append=True, append_new_line=False)
    return errors


@pytest.mark.parametrize(
    ("task"),
    MTEB().tasks_cls,
    ids=lambda task: task.metadata.name,
)
def test_dataset_conforms_to_schema(task: AbsTask):
    if any(
        name in task.metadata.name
        for name in [
            # "DiaBla",
            # "AI-Sweden/SuperLim",
            # "FloresBitextMining",
            # "FloresClusteringS2S",
            # "OpusparcusPC",
            # "GerDaLIR",
            # "GermanQuAD-Retrieval",
            # "SpanishPassageRetrievalS2P",
            # "SpanishPassageRetrievalS2S",
            # "SyntecRetrieval",
            # "AlloprofRetrieval",
            # "XMarket",
            # "NorQuadRetrieval",
            # "ArguAna-PL",
            # "MSMARCOv2",
            # "FiQA-PL",
            # "NarrativeQARetrieval",
            # "SCIDOCS-PL",
            # "NFCorpus",
            # "SciFact-PL",
            # "MSMARCO-PL",
            # "HotpotQA-PL",
            # "VideoRetrieval",
            # "T2Retrieval",
            # "MMarcoRetrieval",
            # "MedicalRetrieval",
            # "DBPedia-PL",
            # "TRECCOVID-PL",
            # "NQ-PL",
            # "CovidRetrieval",
            # "CmedqaRetrieval",
            # "HotpotQA",
            # "DuRetrieval",
            # "EcomRetrieval",
            # "Ko-mrtydi",
            # "Quora-PL",
            # "NFCorpus-PL",
            # "MLSUMClusteringP2P",
            # "SwednClustering",
            # "SwednRetrieval",
            # "NordicLangClassification",
            # "SweFaqRetrieval",
            # "MLSUMClusteringS2S",
            # "DalajClassification",
            # "CQADupstackStatsRetrieval",
            # "Ko-StrategyQA",
            # "AlloProfClusteringS2S",
            # "CQADupstackWebmastersRetrieval",
            # "TRECCOVID",
            # "DanFEVER",
            # "CQADupstackEnglishRetrieval",
            # "CQADupstackGisRetrieval",
            # "SciFact",
            # "Ko-miracl",
            # "AlloProfClusteringP2P",
            # "ArguAna",
            # "CQADupstackWordpressRetrieval",
            # "MSMARCO",
            # "CQADupstackUnixRetrieval",
            # "SCIDOCS",
            # "FiQA2018",
            # "QuoraRetrieval",
            # "CQADupstackTexRetrieval",
            # "BSARDRetrieval",
            # "ClimateFEVER",
            # "GermanDPR",
            # "TwitterHjerneRetrieval",
            # "DBPedia",
            # "Ocnli",
            # "Touche2020",
            # "CQADupstackGamingRetrieval",
            # "CQADupstackPhysicsRetrieval",
            # "CQADupstackProgrammersRetrieval",
            # "TV2Nordretrieval",
            # "CQADupstackAndroidRetrieval",
            # "Cmnli",
            # "NQ",
            # "SNLRetrieval",
            # "FEVER",
            # "HagridRetrieval",
            # "CQADupstackMathematicaRetrieval",
        ]
    ):
        return

    errors = check_task_hf_specification(task)
    if errors:
        pretty_print = "\n".join([f"{err.name} - {err.error}" for err in errors])
        assert False, f"Datasets not available on Hugging Face:\n{pretty_print}"
