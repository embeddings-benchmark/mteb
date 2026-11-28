from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LocBenchReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LocBenchRR",
        description="Software Issue Localization.",
        reference="https://aclanthology.org/2025.acl-long.426/",
        dataset={
            "path": "mteb/LocBenchRR",
            "revision": "4a1fe7be94481b1e5a47d072782c89b146b4fecf",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="recall_at_10",
        date=("2025-03-12", "2025-03-12"),  # arxiv v1 submission date
        domains=["Programming", "Written"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt={
            "query": "Given a github issue, identify the code that needs to be changed to fix the issue."
        },
        bibtex_citation=r"""
@inproceedings{chen2025locagentgraphguidedllmagents,
  address = {Vienna, Austria},
  author = {Chen, Zhaoling and Tang, Robert and Deng, Gangda and Wu, Fang and Wu, Jialong and Jiang, Zhiwei and Prasanna, Viktor and Cohan, Arman and Wang, Xingyao},
  booktitle = {Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.acl-long.426},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-251-0},
  month = {jul},
  pages = {8697--8727},
  publisher = {Association for Computational Linguistics},
  title = {{L}oc{A}gent: Graph-Guided {LLM} Agents for Code Localization},
  url = {https://aclanthology.org/2025.acl-long.426/},
  year = {2025},
}
""",
    )
