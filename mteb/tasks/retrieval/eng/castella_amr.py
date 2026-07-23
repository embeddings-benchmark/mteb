from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CASTELLAAMRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CASTELLAAMRRetrieval",
        description=(
            "Retrieval on CASTELLA, the human-annotated long-audio benchmark used "
            "as the DCASE 2026 Task 6 evaluation set: given a caption describing a "
            "moment, retrieve the full recording (1 to 5 minutes) that contains it. "
            "The corpus is the 566 complete recordings and there are 1,347 caption "
            "queries, each with its source recording as the single relevant "
            "document. Adapted from CASTELLA's audio moment localization task, "
            "which regresses a continuous time interval, to whole-recording "
            "retrieval for the mteb ranking format."
        ),
        reference="https://arxiv.org/abs/2511.15131",
        dataset={
            "path": "dukesun99/CASTELLA-AMR",
            "revision": "083b2816174890f60f36fa9f145cfe79c2f94d0a",
        },
        type="Any2AnyRetrieval",
        category="t2a",
        modalities=["text", "audio"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-11-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{munakata2025castella,
  archiveprefix = {arXiv},
  author = {Hokuto Munakata and Takehiro Imamura and Taichi Nishimura and Tatsuya Komatsu},
  eprint = {2511.15131},
  primaryclass = {eess.AS},
  title = {CASTELLA: Long Audio Dataset with Captions and Temporal Boundaries},
  url = {https://arxiv.org/abs/2511.15131},
  year = {2025},
}
""",
        prompt={"query": "Retrieve the recording that contains the described moment."},
    )
