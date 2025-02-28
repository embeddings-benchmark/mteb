from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeFeedbackMT(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeFeedbackMT",
        description="The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.",
        reference="https://arxiv.org/abs/2402.14658",
        dataset={
            "path": "CoIR-Retrieval/codefeedback-mt",
            "revision": "b0f12fa0c0dd67f59c95a5c33d02aeeb4c398c5f",
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
        bibtex_citation="""@misc{zheng2024opencodeinterpreterintegratingcodegeneration,
              title={OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement},
              author={Tianyu Zheng and Ge Zhang and Tianhao Shen and Xueling Liu and Bill Yuchen Lin and Jie Fu and Wenhu Chen and Xiang Yue},
              year={2024},
              eprint={2402.14658},
              archivePrefix={arXiv},
              primaryClass={cs.SE},
              url={https://arxiv.org/abs/2402.14658},
        }""",
    )
