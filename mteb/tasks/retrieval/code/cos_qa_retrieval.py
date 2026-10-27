from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class CosQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CosQA",
        description="The dataset is a collection of natural language queries and their corresponding code snippets. The task is to retrieve the most relevant code snippet for a given query.",
        reference="https://aclanthology.org/2021.acl-long.442/",
        dataset={
            "path": "CoIR-Retrieval/cosqa",
            "revision": "bc5efb7e9d437246ce393ed19d772e08e4a79535",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-05-07", "2021-05-07"),
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{huang2021cosqa20000webqueries,
  address = {Online},
  author = {Huang, Junjie and Tang, Duyu and Shou, Linjun and Gong, Ming and Xu, Ke and Jiang, Daxin and Zhou, Ming and Duan, Nan},
  booktitle = {Proceedings of the 59th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.442},
  editor = {Zong, Chengqing and Xia, Fei and Li, Wenjie and Navigli, Roberto},
  month = {aug},
  pages = {5690--5700},
  publisher = {Association for Computational Linguistics},
  title = {{C}o{SQA}: 20,000+ Web Queries for Code Search and Question Answering},
  url = {https://aclanthology.org/2021.acl-long.442/},
  year = {2021},
}
""",
    )
