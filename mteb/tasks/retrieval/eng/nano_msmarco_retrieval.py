from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NanoMSMARCORetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoMSMARCORetrieval",
        description="NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.",
        reference="https://microsoft.github.io/msmarco/",
        dataset={
            "path": "mteb/NanoMSMARCORetrieval",
            "revision": "5a5aa0c3df65d6883d85c8f5cdc88e679633595e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2016-01-01", "2016-12-31"],
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{bajaj2018msmarcohumangenerated,
  archiveprefix = {arXiv},
  author = {Payal Bajaj and Daniel Campos and Nick Craswell and Li Deng and Jianfeng Gao and Xiaodong Liu and Rangan Majumder and Andrew McNamara and Bhaskar Mitra and Tri Nguyen and Mir Rosenberg and Xia Song and Alina Stoica and Saurabh Tiwary and Tong Wang},
  eprint = {1611.09268},
  primaryclass = {cs.CL},
  title = {MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},
  url = {https://arxiv.org/abs/1611.09268},
  year = {2018},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
        adapted_from=["MSMARCO"],
    )
