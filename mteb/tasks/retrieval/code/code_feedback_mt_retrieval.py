from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class CodeFeedbackMT(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CodeFeedbackMT",
        description="The dataset is a collection of user queries and assistant responses. The task is to retrieve the most relevant response for a given query.",
        reference="https://aclanthology.org/2024.findings-acl.762/",
        dataset={
            "path": "CoIR-Retrieval/codefeedback-mt",
            "revision": "b0f12fa0c0dd67f59c95a5c33d02aeeb4c398c5f",
        },
        type="Retrieval",
        category="t2t",
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
        bibtex_citation=r"""
@inproceedings{zheng2024opencodeinterpreterintegratingcodegeneration,
  address = {Bangkok, Thailand},
  author = {Zheng, Tianyu and Zhang, Ge and Shen, Tianhao and Liu, Xueling and Lin, Bill Yuchen and Fu, Jie and Chen, Wenhu and Yue, Xiang},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2024},
  doi = {10.18653/v1/2024.findings-acl.762},
  editor = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  month = {aug},
  pages = {12834--12859},
  publisher = {Association for Computational Linguistics},
  title = {{O}pen{C}ode{I}nterpreter: Integrating Code Generation with Execution and Refinement},
  url = {https://aclanthology.org/2024.findings-acl.762/},
  year = {2024},
}
""",
    )
