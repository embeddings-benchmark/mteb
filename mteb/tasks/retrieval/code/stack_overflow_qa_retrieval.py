from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class StackOverflowQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="StackOverflowQA",
        description="The dataset is a collection of natural language queries and their corresponding response which may include some text mixed with code snippets. The task is to retrieve the most relevant response for a given query.",
        reference="https://aclanthology.org/2025.acl-long.1072/",
        dataset={
            "path": "mteb/StackOverflowQA",
            "revision": "7b46f5ebd2365278c1bb025986dda2fee893bbf5",
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
@inproceedings{li2024coircomprehensivebenchmarkcode,
  address = {Vienna, Austria},
  author = {Li, Xiangyang and Dong, Kuicai and Lee, Yi Quan and Xia, Wei and Zhang, Hao and Dai, Xinyi and Wang, Yasheng and Tang, Ruiming},
  booktitle = {Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.acl-long.1072},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-251-0},
  month = {jul},
  pages = {22074--22091},
  publisher = {Association for Computational Linguistics},
  title = {{C}o{IR}: A Comprehensive Benchmark for Code Information Retrieval Models},
  url = {https://aclanthology.org/2025.acl-long.1072/},
  year = {2025},
}
""",
    )
