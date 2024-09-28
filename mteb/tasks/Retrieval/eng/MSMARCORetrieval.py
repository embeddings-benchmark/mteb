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
        eval_splits=["train", "dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "train": {
                    "average_document_length": 335.79716603691344,
                    "average_query_length": 33.21898281898998,
                    "num_documents": 8841823,
                    "num_queries": 502939,
                    "average_relevant_docs_per_query": 1.0592755781516248,
                },
                "dev": {
                    "average_document_length": 335.79716603691344,
                    "average_query_length": 33.2621776504298,
                    "num_documents": 8841823,
                    "num_queries": 6980,
                    "average_relevant_docs_per_query": 1.0654727793696275,
                },
                "test": {
                    "average_document_length": 335.79716603691344,
                    "average_query_length": 32.74418604651163,
                    "num_documents": 8841823,
                    "num_queries": 43,
                    "average_relevant_docs_per_query": 95.3953488372093,
                },
            },
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
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
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
        descriptive_stats={
            "n_samples": {"test": 43},
            "avg_character_length": {
                "test": {
                    "average_document_length": 355.2909668633681,
                    "average_query_length": 32.74418604651163,
                    "num_documents": 8812,
                    "num_queries": 43,
                    "average_relevant_docs_per_query": 95.3953488372093,
                }
            },
        },
    )
