from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CoVRRVT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CoVRRVT2VRetrieval",
        description=(
            "CoVR-R is a reasoning-aware benchmark for composed video retrieval. "
            "Given a reference video and a textual modification, the goal is to "
            "retrieve the correct target video that reflects the requested change "
            "and its implied visual consequences."
        ),
        reference="https://arxiv.org/abs/2603.20190",
        dataset={
            "path": "whybe-choi/CoVRRVT2VRetrieval",
            "revision": "c4c64ba85adf72c1577fcb693dee0b1aa1ac8080",
        },
        type="Any2AnyRetrieval",
        category="vt2v",
        modalities=["video", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_5",
        date=("2026-03-20", "2026-03-20"),
        domains=["Web", "Activity"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="apache-2.0",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{thawakar2026covrr,
  author = {Thawakar, Omkar and Demidov, Dmitry and Potlapalli, Vaishnav and Bogireddy, Sai Prasanna Teja Reddy and Gajjala, Viswanatha Reddy and Lasheen, Alaa Mostafa and Anwer, Rao Muhammad and Khan, Fahad Shahbaz},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  title = {CoVR-R: Reason-Aware Composed Video Retrieval},
  year = {2026},
}
""",
    )
