from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FineCVRVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FineCVRVT2VRetrieval",
        description=(
            "FineCVR is a fine-grained composed video retrieval benchmark. "
            "Given a reference video and a textual modification, the goal is to "
            "retrieve the target video that reflects the requested change."
        ),
        reference="https://github.com/May2333/FDCA",
        dataset={
            "path": "myang333/FineCVRVT2VRetrieval",
            "revision": "24e057a45717a3c62fe7e7a87232379ab8d75327",
        },
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=("2025-02-22", "2025-02-22"),
        domains=["Web", "Activity"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{yue25finecvr,
  title = {Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval},
  author = {Yue Wu and Zhaobo Qi and Yiling Wu and Junshu Sun and Yaowei Wang and Shuhui Wang},
  journal = {ICLR},
  year = {2025},
}
""",
        is_beta=True,
    )
