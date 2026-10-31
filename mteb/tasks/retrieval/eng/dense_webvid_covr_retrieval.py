from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class DenseWebVidCoVRVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DenseWebVidCoVRVT2VRetrieval",
        description=(
            "Dense-WebVid-CoVR is a dataset for Compositional Video Retrieval (CoVR) "
            "where the query consists of a reference video and an editing instruction. "
            "The corpus consists of candidate videos."
        ),
        reference="https://doi.org/10.1109/iccv51701.2025.01900",
        dataset={
            "path": "nik1995/Dense-WebVid-CoVR",
            "revision": "988f558c4497429f98b2b16789863aba403449d7",
        },
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2024-12-31"),
        domains=["Web"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{thawakar2025simpleeditscomposedvideo,
  author = {Thawakar, Omkar and Demidov, Dmitry and Thawkar, Ritesh and Anwer, Rao Muhammad and Shah, Mubarak and Khan, Fahad Shahbaz and Khan, Salman},
  booktitle = {2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  doi = {10.1109/iccv51701.2025.01900},
  month = {Oct},
  pages = {20435–20444},
  publisher = {IEEE},
  title = {Beyond Simple Edits: Composed Video Retrieval with Dense Modifications},
  url = {https://doi.org/10.1109/iccv51701.2025.01900},
  year = {2025},
}
""",
        prompt={
            "query": "Given the reference video and editing text, retrieve the video that matches the composed query."
        },
        is_beta=True,
    )
