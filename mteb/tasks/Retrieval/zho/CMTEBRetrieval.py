from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class T2Retrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="T2Retrieval",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        dataset={
            "path": "mteb/T2Retrieval",
            "revision": "cf778c0ea4168ec5174a34d888d6453e4cde9222",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2023-04-04", "2023-05-16"),
        domains=[
            "Medical",
            "Academic",
            "Financial",
            "Government",
            "Non-fiction",
        ],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{xie2023t2ranking,
  archiveprefix = {arXiv},
  author = {Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
  eprint = {2304.03679},
  primaryclass = {cs.IR},
  title = {T2Ranking: A large-scale Chinese Benchmark for Passage Ranking},
  year = {2023},
}
""",
        prompt={
            "query": "Given a Chinese search query, retrieve web passages that answer the question"
        },
    )


class MMarcoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MMarcoRetrieval",
        description="MMarcoRetrieval",
        reference="https://arxiv.org/abs/2309.07597",
        dataset={
            "path": "mteb/MMarcoRetrieval",
            "revision": "4940a7b26bf53463cfe3435bb8e201963e9c31ae",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{xiao2024cpack,
  archiveprefix = {arXiv},
  author = {Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
  eprint = {2309.07597},
  primaryclass = {cs.CL},
  title = {C-Pack: Packaged Resources To Advance General Chinese Embedding},
  year = {2024},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )


class DuRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DuRetrieval",
        description="A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        dataset={
            "path": "mteb/DuRetrieval",
            "revision": "313c81b51311893c8fd09ca432f96b841ed0ebb3",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{qiu2022dureaderretrieval,
  archiveprefix = {arXiv},
  author = {Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
  eprint = {2203.10232},
  primaryclass = {cs.CL},
  title = {DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
  year = {2022},
}
""",
        prompt={
            "query": "Given a Chinese search query, retrieve web passages that answer the question"
        },
    )


class CovidRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CovidRetrieval",
        description="COVID-19 news articles",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/CovidRetrieval",
            "revision": "9c6dc4b276bb47c3ff725bbc5ffcafd56dded38b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=("2022-03-03", "2022-03-18"),
        domains=["Medical", "Entertainment"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation=None,
        bibtex_citation=r"""
@misc{long2022multicprmultidomainchinese,
  archiveprefix = {arXiv},
  author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
  eprint = {2203.03367},
  primaryclass = {cs.IR},
  title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
  url = {https://arxiv.org/abs/2203.03367},
  year = {2022},
}
""",
        prompt={
            "query": "Given a question on COVID-19, retrieve news articles that answer the question"
        },
    )


class CmedqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CmedqaRetrieval",
        description="Online medical consultation text. Used the CMedQAv2 as its underlying dataset.",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        dataset={
            "path": "mteb/CmedqaRetrieval",
            "revision": "c476f85bf03d6642ec66bf54b9a551c88108bbb4",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{qiu2022dureaderretrievallargescalechinesebenchmark,
  archiveprefix = {arXiv},
  author = {Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
  eprint = {2203.10232},
  primaryclass = {cs.CL},
  title = {DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
  url = {https://arxiv.org/abs/2203.10232},
  year = {2022},
}
""",
        prompt={
            "query": "Given a Chinese community medical question, retrieve replies that best answer the question"
        },
        adapted_from=["CMedQAv2-reranking"],
    )


class EcomRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="EcomRetrieval",
        description="EcomRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/EcomRetrieval",
            "revision": "fa705ce5418e91636b1eaeaf43f34c15aa3f5a8a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{long2022multicprmultidomainchinese,
  archiveprefix = {arXiv},
  author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
  eprint = {2203.03367},
  primaryclass = {cs.IR},
  title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
  url = {https://arxiv.org/abs/2203.03367},
  year = {2022},
}
""",
        prompt={
            "query": "Given a user query from an e-commerce website, retrieve description sentences of relevant products"
        },
    )


class MedicalRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MedicalRetrieval",
        description="MedicalRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/MedicalRetrieval",
            "revision": "023ae3b2c6b96f583c4ff9b3f9239c93f7885c20",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{long2022multicprmultidomainchinese,
  archiveprefix = {arXiv},
  author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
  eprint = {2203.03367},
  primaryclass = {cs.IR},
  title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
  url = {https://arxiv.org/abs/2203.03367},
  year = {2022},
}
""",
        prompt={
            "query": "Given a medical question, retrieve user replies that best answer the question"
        },
    )


class VideoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="VideoRetrieval",
        description="VideoRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/VideoRetrieval",
            "revision": "146a9d5e4fd7a9c182b6b92cccb6a3753994305c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{long2022multicprmultidomainchinese,
  archiveprefix = {arXiv},
  author = {Dingkun Long and Qiong Gao and Kuan Zou and Guangwei Xu and Pengjun Xie and Ruijie Guo and Jian Xu and Guanjun Jiang and Luxi Xing and Ping Yang},
  eprint = {2203.03367},
  primaryclass = {cs.IR},
  title = {Multi-CPR: A Multi Domain Chinese Dataset for Passage Retrieval},
  url = {https://arxiv.org/abs/2203.03367},
  year = {2022},
}
""",
        prompt={
            "query": "Given a video search query, retrieve the titles of relevant videos"
        },
    )
