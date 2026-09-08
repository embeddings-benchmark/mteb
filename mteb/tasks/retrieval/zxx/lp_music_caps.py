from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

LP_MUSIC_CAPS_CITATION = r"""
@inproceedings{doh2023lpmusiccaps,
  author = {SeungHeon Doh and Keunwoo Choi and Jongpil Lee and Juhan Nam},
  booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR)},
  eprint = {2307.16372},
  title = {LP-MusicCaps: LLM-Based Pseudo Music Captioning},
  url = {https://arxiv.org/abs/2307.16372},
  year = {2023},
}
"""

LP_MUSIC_CAPS_DESCRIPTION = (
    "LLM-generated pseudo captions for 10-second music clips from the MagnaTagATune "
    "dataset. Captions were produced by prompting a large language model with the "
    "human-annotated tags of each clip, giving four differently-styled captions per "
    "clip. Complements MusicCaps, whose captions are human-written and whose audio "
    "comes from AudioSet."
)


class LPMusicCapsMTTA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LPMusicCapsMTTA2TRetrieval",
        description=LP_MUSIC_CAPS_DESCRIPTION,
        reference="https://arxiv.org/abs/2307.16372",
        dataset={
            "path": "hubxrt/LPMusicCapsMTT_a2t",
            "revision": "062eff5c2454c5765ee3f499cde8de78626fe06f",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        # audio is found (MagnaTagATune); captions are LM-generated
        sample_creation="found",
        bibtex_citation=LP_MUSIC_CAPS_CITATION,
    )


class LPMusicCapsMTTT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LPMusicCapsMTTT2ARetrieval",
        description=LP_MUSIC_CAPS_DESCRIPTION,
        reference="https://arxiv.org/abs/2307.16372",
        dataset={
            "path": "hubxrt/LPMusicCapsMTT_t2a",
            "revision": "33c5efd45927bcf67cda707d33365387571c7054",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Music"],
        task_subtypes=["Music Caption Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        # audio is found (MagnaTagATune); captions are LM-generated
        sample_creation="found",
        bibtex_citation=LP_MUSIC_CAPS_CITATION,
    )
