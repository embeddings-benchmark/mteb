from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCOFast(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MSMARCO-Fast",
        dataset={
            "path": "mteb/MSMARCO_test_top_250_only_w_correct",
            "revision": "5d3d15761bd3e6d0885f01e10237c1cb13eb6370",
        },
        description="MS MARCO is a collection of datasets focused on deep learning in search",
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
                    "average_document_length": 354.2790134664851,
                    "average_query_length": 32.74418604651163,
                    "num_documents": 6609,
                    "num_queries": 43,
                    "average_relevant_docs_per_query": 48.02325581395349,
                }
            },
        },
    )
