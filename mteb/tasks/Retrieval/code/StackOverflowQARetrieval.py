from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class StackOverflowQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StackOverflowQA",
        description="The dataset is a collection of natural language queries and their corresponding response which may include some text mixed with code snippets. The task is to retrieve the most relevant response for a given query.",
        reference="https://arxiv.org/abs/2407.02883",
        dataset={
            "path": "CoIR-Retrieval/stackoverflow-qa",
            "revision": "db8f169f3894c14a00251061f957b2063eef2bd5",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2024coircomprehensivebenchmarkcode,
        title={CoIR: A Comprehensive Benchmark for Code Information Retrieval Models},
        author={Xiangyang Li and Kuicai Dong and Yi Quan Lee and Wei Xia and Yichun Yin and Hao Zhang and Yong Liu and Yasheng Wang and Ruiming Tang},
        year={2024},
        eprint={2407.02883},
        archivePrefix={arXiv},
        primaryClass={cs.IR},
        url={https://arxiv.org/abs/2407.02883},
        }""",
        descriptive_stats={
            "n_samples": {
                _EVAL_SPLIT: 1000,
            },
            "avg_character_length": {
                "test": {
                    "average_document_length": 1202.4815613867845,
                    "average_query_length": 1302.6263791374122,
                    "num_documents": 19931,
                    "num_queries": 1994,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
