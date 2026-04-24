"""Test if the metadata of all tasks is filled and valid."""

import pytest

from mteb.abstasks import AbsTask, AbsTaskRetrieval
from mteb.get_tasks import get_tasks

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
# Tasks should be removed from this list once their metadata is filled.
_HISTORIC_DATASETS = []

_EXISTING_VIDEO_TASKS = [
    "BreakfastClassification",
    "HMDB51Classification",
    "Kinetics400VA",
    "Kinetics400V",
    "SomethingSomethingV2Classification",
    "VGGSoundVA",
    "VGGSoundV",
    "AVEDatasetClustering",
    "MusicAVQAClustering",
    "RAVDESSAVClustering",
    "WorldSense1MinDomainClustering",
    "MSRVTTT2VA",
    "MSRVTTVA2T",
    "ActivityNetCaptionsT2VRetrieval",
    "ActivityNetCaptionsV2TRetrieval",
    "DiDeMoT2VARetrieval",
    "DiDeMoT2VRetrieval",
    "DiDeMoV2TRetrieval",
    "DiDeMoVA2TRetrieval",
    "MSVDT2VRetrieval",
    "MSVDV2TRetrieval",
    "Shot2Story20KT2VARetrieval",
    "Shot2Story20KT2VRetrieval",
    "Shot2Story20KV2TRetrieval",
    "Shot2Story20KVA2TRetrieval",
    "TUNABenchT2VRetrieval",
    "TUNABenchV2TRetrieval",
    "VATEXT2VARetrieval",
    "VATEXT2VRetrieval",
    "VATEXV2TRetrieval",
    "VATEXVA2TRetrieval",
    "YouCook2T2VARetrieval",
    "YouCook2T2VRetrieval",
    "YouCook2V2TRetrieval",
    "YouCook2VA2TRetrieval",
    "Kinetics400ZeroShot",
    "UCF101Clustering",
    "VALOR32KT2VARetrieval",
    "VALOR32KT2VRetrieval",
    "VALOR32KV2TRetrieval",
    "VALOR32KVA2TRetrieval",
    "HMDB51Clustering",
    "AVMemeExamT2VARetrieval",
    "AVMemeExamT2VRetrieval",
    "AVMemeExamV2TRetrieval",
    "AVMemeExamVA2TRetrieval",
    "AVMemeExamVA2TRetrieval",
    "AudioCapsAVT2VARetrieval",
    "AudioCapsAVT2VRetrieval",
    "AudioCapsAVV2TRetrieval",
    "AudioCapsAVVA2TRetrieval",
    "Panda70MT2VARetrieval",
    "Panda70MT2VRetrieval",
    "Panda70MV2TRetrieval",
    "Panda70MVA2TRetrieval",
    "VGGSoundAVT2VARetrieval",
    "VGGSoundAVT2VRetrieval",
    "VGGSoundAVV2TRetrieval",
    "VGGSoundAVVA2TRetrieval",
]


@pytest.mark.parametrize(
    "task",
    get_tasks(exclude_superseded=False, exclude_aggregate=False, exclude_beta=False),
    ids=lambda x: x.metadata.name,
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

    # --- Check simplifiedtasktypes is valid ---
    assert isinstance(task.metadata.simplified_task_type, str)

    # --- Test is descriptive stats are present for all datasets ---
    if task.is_aggregate:  # aggregate tasks do not have descriptive stats
        return

    # TODO https://github.com/embeddings-benchmark/mteb/issues/3498
    if task.metadata.name in (  # noqa: PLR6201
        "FleursA2TRetrieval",
        "FleursT2ARetrieval",
        "SoundDescsA2TRetrieval",
        "SoundDescsT2ARetrieval",
        "BirdSet",
        "AudioSet",
    ):
        assert task.metadata.descriptive_stats is None
        pytest.skip("Skipping audio tasks for now, see issue #3498")

    # TODO https://github.com/embeddings-benchmark/mteb/issues/4378
    if task.metadata.name in _EXISTING_VIDEO_TASKS:
        assert task.metadata.descriptive_stats is None
        pytest.skip("Skipping video tasks for now, see issue #4378")

    assert task.metadata.descriptive_stats is not None, (
        f"Dataset {task.metadata.name} should have descriptive stats. You can add metadata to your task by running `YourTask().calculate_descriptive_statistics()`"
    )
    assert task.metadata.n_samples is not None

    if task.metadata.prompt is not None and isinstance(task.metadata.prompt, dict):
        if not (
            isinstance(task, AbsTaskRetrieval) or task.metadata.name in ["TERRa.V2"]  # noqa: PLR6201
        ):
            # Retrieval tasks and TERRa.V2 have a dict prompt, but other tasks should not
            raise ValueError(
                f"Task {task.metadata.name} has a dict prompt, but it should be a string. Please check the metadata of the task."
            )
