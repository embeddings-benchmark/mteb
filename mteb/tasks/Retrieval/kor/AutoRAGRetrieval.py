from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AutoRAGRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AutoRAGRetrieval",
        description="Korean Retrieval Task origniated from AutoRAG",
        reference="https://arxiv.org/abs/2410.20878",
        dataset={
            "path": "yjoonjang/markers_bm",
            "revision": "fd7df84ac089bbec763b1c6bb1b56e985df5cc5c",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2024-08-03", "2024-08-03"),
        domains=["Government", "Medical", "Legal", "Social"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{kim2024autoragautomatedframeworkoptimization,
      title={AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline}, 
      author={Dongkyu Kim and Byoungwook Kim and Donggeon Han and Matouš Eibich},
      year={2024},
      eprint={2410.20878},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.20878}, 
}""",
        descriptive_stats={
            "avg_character_length": {
                "test": {
                    "average_document_length": 983.8421052631579,
                    "average_query_length": 69.6140350877193,
                    "num_documents": 114,
                    "num_queries": 114,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
