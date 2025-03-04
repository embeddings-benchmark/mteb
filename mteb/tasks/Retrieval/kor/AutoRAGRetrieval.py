from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AutoRAGRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AutoRAGRetrieval",
        description="This dataset enables the evaluation of Korean RAG performance across various domains—finance, public sector, healthcare, legal, and commerce—by providing publicly accessible documents, questions, and answers.",
        reference="https://arxiv.org/abs/2410.20878",
        dataset={
            "path": "yjoonjang/markers_bm",
            "revision": "fd7df84ac089bbec763b1c6bb1b56e985df5cc5c",
        },
        type="Retrieval",
        prompt="Retrieve text based on user query.",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2024-08-03", "2024-08-03"),
        domains=["Government", "Medical", "Legal", "Social", "Financial"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@misc{kim2024autoragautomatedframeworkoptimization,
      title={AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline},
      author={Dongkyu Kim and Byoungwook Kim and Donggeon Han and Matouš Eibich},
      year={2024},
      eprint={2410.20878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.20878},
}""",
    )
