from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MMEditAT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMEditAT2ARetrieval",
        description=(
            "Composed audio retrieval derived from the MMEdit test set. Each query "
            "combines an unedited source recording with a natural-language editing "
            "instruction, and the goal is to retrieve the corresponding edited "
            "recording from a global corpus of 3,317 candidates. Byte-identical "
            "target recordings are all marked relevant. Because edited recordings "
            "usually preserve most of their source audio, source-only matching is a "
            "strong shortcut on this task; results should therefore be interpreted "
            "together with an audio-only diagnostic."
        ),
        reference="https://arxiv.org/abs/2512.20339",
        dataset={
            "path": "pranitchawla/MMEdit-AT2A",
            "revision": "ab1ce44e7fc3dc31292c025b6e002915538d4feb",
        },
        type="Any2AnyRetrieval",
        category="at2a",
        modalities=["audio", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2025-12-23"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{tao2025mmedit,
  author = {Tao, Ye and Xu, Xuenan and Wu, Wen and Wang, Shuai and Wu, Mengyue and Zhang, Chao},
  journal = {arXiv preprint arXiv:2512.20339},
  title = {MMEDIT: A Unified Framework for Multi-Type Audio Editing via Audio Language Model},
  url = {https://arxiv.org/abs/2512.20339},
  year = {2025},
}
""",
        prompt={
            "query": "Given the source audio and an editing instruction, retrieve the corresponding edited audio."
        },
        is_beta=True,
    )
