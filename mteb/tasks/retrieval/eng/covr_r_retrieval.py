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
@misc{thawakar2026covrrreasonawarecomposedvideoretrieval,
  archiveprefix = {arXiv},
  author = {Omkar Thawakar and Dmitry Demidov and Vaishnav Potlapalli and Sai Prasanna Teja Reddy Bogireddy and Viswanatha Reddy Gajjala and Alaa Mostafa Lasheen and Rao Muhammad Anwer and Fahad Khan},
  eprint = {2603.20190},
  primaryclass = {cs.CV},
  title = {CoVR-R:Reason-Aware Composed Video Retrieval},
  url = {https://arxiv.org/abs/2603.20190},
  year = {2026},
}
""",
    )
