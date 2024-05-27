from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WebLINXCandidatesReranking",
        description="WEBLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://mcgill-nlp.github.io/weblinx",
        dataset={
            "path": "McGill-NLP/WebLINX",  # TODO: change to the actual path
            "name": "reranking",
            "revision": "4d5db43bcd72c036b8f8dd0740d9ed2cee71d09b",  # TODO: change to commit hash
        },
        type="Reranking",
        category="p2p",
        eval_splits=[
            "validation",
            "test_iid",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2023-03-01", "2023-10-30"),
        form=["written"],
        domains=["Academic", "Web"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
@misc{lù2024weblinx,
      title={WebLINX: Real-World Website Navigation with Multi-Turn Dialogue}, 
      author={Xing Han Lù and Zdeněk Kasner and Siva Reddy},
      year={2024},
      eprint={2402.05930},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
        """,
        n_samples={
            "validation": 1301,
            "test_iid": 1438,
            "test_cat": 3560,
            "test_web": 3144,
            "test_vis": 5298,
            "test_geo": 4916,
        },
        avg_character_length={
            "validation": 1647.52,
            "test_iid": 1722.63,
            "test_cat": 2149.66,
            "test_web": 1831.46,
            "test_vis": 1737.26,
            "test_geo": 1742.66,
        },
    )
