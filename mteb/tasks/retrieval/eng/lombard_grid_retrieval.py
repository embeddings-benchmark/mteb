from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LombardGridI2VARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LombardGridI2VARetrieval",
        description=(
            "Image-to-video+audio speaker retrieval derived from the Lombard GRID "
            "corpus. The 540 queries are midpoint frames from frontal videos, and the "
            "1,080 corpus items pair a profile-view video with the separate audio from "
            "the same recording. Relevance is derived from speaker identity: each "
            "query has all 20 corpus recordings from the same speaker as positives. "
            "Query and corpus recordings use disjoint utterance codes and camera "
            "views, and both sets are balanced between plain and Lombard speech. The "
            "source paper introduced the corpus, not this retrieval benchmark."
        ),
        reference="https://doi.org/10.1121/1.5042758",
        dataset={
            "path": "Cerru02/LombardGrid-I2VA",
            "revision": "9c3109684a2454e1c9ee0410c924efaae90a46f7",
        },
        type="Any2AnyRetrieval",
        category="i2va",
        modalities=["image", "video", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        # The source does not report collection dates; use the publication year.
        date=("2018-01-01", "2018-12-31"),
        domains=["Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=["eng-GB"],
        sample_creation="created",
        is_beta=True,
        bibtex_citation=r"""
@article{alghamdi2018corpus,
  author = {Alghamdi, Najwa and Maddock, Steve and Marxer, Ricard and Barker, Jon and Brown, Guy J.},
  doi = {10.1121/1.5042758},
  journal = {The Journal of the Acoustical Society of America},
  number = {6},
  pages = {EL523--EL529},
  title = {A corpus of audio-visual Lombard speech with frontal and profile views},
  volume = {143},
  year = {2018},
}
""",
        prompt={
            "query": (
                "Retrieve profile-view video and audio recordings of the same speaker "
                "shown in the frontal image."
            )
        },
    )
