from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
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
        bibtex_citation="""@article{DBLP:journals/corr/NguyenRSGTMD16,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  journal   = {CoRR},
  volume    = {abs/1611.09268},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.09268},
  archivePrefix = {arXiv},
  eprint    = {1611.09268},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
}""",
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
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
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
        bibtex_citation="""@article{DBLP:journals/corr/NguyenRSGTMD16,
  author    = {Tri Nguyen and
               Mir Rosenberg and
               Xia Song and
               Jianfeng Gao and
               Saurabh Tiwary and
               Rangan Majumder and
               Li Deng},
  title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  journal   = {CoRR},
  volume    = {abs/1611.09268},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.09268},
  archivePrefix = {arXiv},
  eprint    = {1611.09268},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
}""",
        adapted_from=["MSMARCO"],
    )
