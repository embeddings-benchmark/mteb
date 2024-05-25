from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WebLINXCandidatesReranking",
        description="WEBLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://mcgill-nlp.github.io/weblinx",
        dataset={
            "path": "https://huggingface.co/datasets/xhluca/WebLINX-testing",  # TODO: change to the actual path
            "revision": "main", # TODO: change to commit hash
        },
        type="Reranking",
        category="p2p",
        eval_splits=["validation", "test_iid", "test_cat", "test_geo", "test_vis", "test_web"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2023-03-01", "2023-10-30"),
        form=['written'],
        domains=['Academic', 'Web'],
        task_subtypes=['Code retrieval', 'Conversational retrieval'],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators='export-annotated',
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
        n_samples={"validation": None, "test_iid": None, "test_cat": None, "test_geo": None, "test_vis": None, "test_web": None},
        avg_character_length={"validation": None, "test_iid": None, "test_cat": None, "test_geo": None, "test_vis": None, "test_web": None},
    )
