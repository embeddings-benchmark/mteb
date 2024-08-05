from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeTransOceanDLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeTransOceanDL",
        description="The dataset is a collection of code snippets and their corresponding natural language queries. The task is to retrieve the most relevant code snippet",
        reference="https://arxiv.org/abs/2310.04951",
        dataset={
            "path": "mteb/codetrans-dl",
            "revision": "ced14bd1adc47a2f83dd99228d5088d4fd6676c0",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-10-08", "2023-10-08"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="Apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{yan2023codetransoceancomprehensivemultilingualbenchmark,
              title={CodeTransOcean: A Comprehensive Multilingual Benchmark for Code Translation}, 
              author={Weixiang Yan and Yuchen Tian and Yunzhe Li and Qian Chen and Wen Wang},
              year={2023},
              eprint={2310.04951},
              archivePrefix={arXiv},
              primaryClass={cs.AI},
              url={https://arxiv.org/abs/2310.04951}, 
        }""",
        descriptive_stats={
            "avg_character_length": {
                "test": {
                    "average_document_length": 1479.0735294117646,
                    "average_query_length": 1867.6222222222223,
                    "num_documents": 816,
                    "num_queries": 180,
                    "average_relevant_docs_per_query": 1.0,
                }
            }
        },
    )
