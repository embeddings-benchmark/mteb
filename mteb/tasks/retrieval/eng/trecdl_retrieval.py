from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TRECDL2019(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="TRECDL2019",
        dataset={
            "path": "whybe-choi/trec-dl-2019",
            "revision": "cbb055c2528c544edf84db8ec6aef1885ec8cfda",
        },
        description="TREC Deep Learning Track 2019 passage ranking task. The task involves retrieving relevant passages from the MS MARCO collection given web search queries. Queries have multi-graded relevance judgments.",
        reference="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
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
@inproceedings{craswell2020overview,
  author = {Craswell, Nick and Mitra, Bhaskar and Yilmaz, Emine and Campos, Daniel and Voorhees, Ellen M},
  booktitle = {Proceedings of the 28th Text REtrieval Conference (TREC 2019)},
  organization = {NIST},
  title = {Overview of the TREC 2019 deep learning track},
  year = {2020},
}

@article{DBLP:journals/corr/NguyenRSGTMD16,
  archiveprefix = {arXiv},
  author = {Tri Nguyen and
Mir Rosenberg and
Xia Song and
Jianfeng Gao and
Saurabh Tiwary and
Rangan Majumder and
Li Deng},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  eprint = {1611.09268},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  url = {http://arxiv.org/abs/1611.09268},
  volume = {abs/1611.09268},
  year = {2016},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )


class TRECDL2020(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="TRECDL2020",
        dataset={
            "path": "whybe-choi/trec-dl-2020",
            "revision": "83d7d3532f3fa653f132da188bbaa852d959d6f9",
        },
        description="TREC Deep Learning Track 2020 passage ranking task. The task involves retrieving relevant passages from the MS MARCO collection given web search queries. Queries have multi-graded relevance judgments.",
        reference="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
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
@inproceedings{craswell2021overview,
  author = {Craswell, Nick and Mitra, Bhaskar and Yilmaz, Emine and Campos, Daniel and Voorhees, Ellen M},
  booktitle = {Proceedings of the 29th Text REtrieval Conference (TREC 2020)},
  organization = {NIST},
  title = {Overview of the TREC 2020 deep learning track},
  year = {2021},
}

@article{DBLP:journals/corr/NguyenRSGTMD16,
  archiveprefix = {arXiv},
  author = {Tri Nguyen and
Mir Rosenberg and
Xia Song and
Jianfeng Gao and
Saurabh Tiwary and
Rangan Majumder and
Li Deng},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  eprint = {1611.09268},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  url = {http://arxiv.org/abs/1611.09268},
  volume = {abs/1611.09268},
  year = {2016},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )
