from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class CosQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CosQA",
        description="The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://arxiv.org/abs/2105.13239",
        dataset={
            "path": "mteb/cosqa",
            "revision": "60b9d6532d06381c6e1fa54fde4f6c86a47dee3a",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-05-07", "2021-05-07"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="MIT",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{huang2021cosqa20000webqueries,
              title={CoSQA: 20,000+ Web Queries for Code Search and Question Answering}, 
              author={Junjie Huang and Duyu Tang and Linjun Shou and Ming Gong and Ke Xu and Daxin Jiang and Ming Zhou and Nan Duan},
              year={2021},
              eprint={2105.13239},
              archivePrefix={arXiv},
              primaryClass={cs.CL},
              url={https://arxiv.org/abs/2105.13239}, 
        }""",
        descriptive_stats={
            "avg_character_length": {
                "test": {
                    "average_document_length": 276.132741215298,
                    "average_query_length": 36.814,
                    "num_documents": 20604,
                    "num_queries": 500,
                    "average_relevant_docs_per_query": 1.0,
                }
            }
        },
    )
