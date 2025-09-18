from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class SWEbenchLiteReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SWEbenchLiteRR",
        description="Software Issue Localization.",
        reference="https://www.swebench.com/",
        dataset={
            "path": "tarsur909/mteb-swe-bench-lite-reranking",
            "revision": "9020779825304b569312509a068219d1771bae7d",
        },
        type="Reranking",
        category="p2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="recall_at_10",
        date=("2023-10-10", "2023-10-10"), 
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Instruct: Given a github issue, identify the code that needs to be changed to fix the issue. Query: ",
        bibtex_citation=r"""
@misc{jimenez2024swebenchlanguagemodelsresolve,
      title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?}, 
      author={Carlos E. Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik Narasimhan},
      year={2024},
      eprint={2310.06770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.06770}, 
}
"""
    )
