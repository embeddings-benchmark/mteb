from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_CITATION = r"""
@inproceedings{chen2020vggsound,
  author = {Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle = {ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  organization = {IEEE},
  title = {Vggsound: A Large-Scale Audio-Visual Dataset},
  year = {2020},
}
"""


class VGGSoundA2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VGGSoundA2VRetrieval",
        description=(
            "Audio-to-video retrieval on VGGSound: given an audio clip, retrieve "
            "the matching video. Packaged from 11hu83/vggsound with a seeded "
            "subsample of 2048 pairs (seed=42). Distinct from the caption-aligned "
            "VGGSoundAV*Retrieval tasks (~696 examples on mteb/VGGSound_AV_RETRIEVAL)."
        ),
        reference="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        dataset={
            "path": "Wissam42/VGGSound-A2V",
            "revision": "af22db7665303d8a0df05e59ffa5e6258a74e0c8",
        },
        type="Any2AnyRetrieval",
        category="a2v",
        modalities=["audio", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION,
        prompt={"query": "Find the video that corresponds to the following audio."},
        is_beta=True,
    )


class VGGSoundV2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VGGSoundV2ARetrieval",
        description=(
            "Video-to-audio retrieval on VGGSound: given a video clip, retrieve "
            "the matching audio. Packaged from 11hu83/vggsound with a seeded "
            "subsample of 2048 pairs (seed=42). Distinct from the caption-aligned "
            "VGGSoundAV*Retrieval tasks (~696 examples on mteb/VGGSound_AV_RETRIEVAL)."
        ),
        reference="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        dataset={
            "path": "Wissam42/VGGSound-V2A",
            "revision": "9fea030aac4526abe85bb99d6ab210f3c25ddc4c",
        },
        type="Any2AnyRetrieval",
        category="v2a",
        modalities=["video", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["AudioScene", "Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_CITATION,
        prompt={"query": "Find the audio that corresponds to the following video."},
        is_beta=True,
    )
