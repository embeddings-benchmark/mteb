from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MSMARCO(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO",
        dataset={
            "path": "mteb/msmarco",
            "revision": "c5a29a104738b98a9e76336939199e264163d4a0",
        },
        description="MS MARCO is a collection of datasets focused on deep learning in search",
        reference="https://microsoft.github.io/msmarco/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
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
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )


class MSMARCOHardNegatives(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCOHardNegatives",
        dataset={
            "path": "mteb/MSMARCO_test_top_250_only_w_correct-v2",
            "revision": "67c0b4f7f15946e0b15cf6cf3b8993d04cb3efc6",
        },
        description="MS MARCO is a collection of datasets focused on deep learning in search. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://microsoft.github.io/msmarco/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
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
