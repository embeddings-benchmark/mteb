from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CASTELLAAMRRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CASTELLAAMRRetrieval",
        description=(
            "Audio moment retrieval on CASTELLA, the human-annotated long-audio "
            "benchmark used as the DCASE 2026 Task 6 evaluation set: given a "
            "caption, retrieve the 10-second window of a long recording (1 to 5 "
            "minutes) that contains the described moment. The corpus holds 12,046 "
            "windows from 566 recordings; 1,347 caption queries with relevance "
            "from at least 50 percent overlap between window and annotated "
            "moment. First temporal-localization retrieval task in the audio "
            "benchmark."
        ),
        reference="https://arxiv.org/abs/2511.15131",
        dataset={
            "path": "dukesun99/CASTELLA-AMR",
            "revision": "6b2a9905a1d0e787e84f635f4f9408941a384883",
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
        prompt={
            "query": "Retrieve the audio segment during which the described moment occurs."
        },
    )
