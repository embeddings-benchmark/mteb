from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MSMARCOv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSMARCOv2",
        dataset={
            "path": "mteb/msmarco-v2",
            "revision": "b1663124850d305ab7c470bb0548acf8e2e7ea43",
        },
        description="MS MARCO is a collection of datasets focused on deep learning in search. This version is derived from BEIR",
        reference="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train", "dev", "dev2"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2016-01-01", "2016-12-31"),  # publication year
        domains=[
            "Encyclopaedic",
            "Academic",
            "Blog",
            "News",
            "Medical",
            "Government",
            "Reviews",
            "Non-fiction",
            "Social",
            "Web",
        ],
        task_subtypes=["Question answering"],
        license="msr-la-nc",
        annotations_creators="derived",
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
        adapted_from=["MSMARCO"],
    )
