from __future__ import annotations

from collections import defaultdict

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(dataset_path, dataset_revision, qrel_revision, eval_splits):
    eval_split = eval_splits[0]
    dataset = load_dataset(dataset_path, revision=dataset_revision)
    qrels = load_dataset(dataset_path + "-qrels", revision=qrel_revision)[eval_split]

    corpus = {e["id"]: {"text": e["text"]} for e in dataset["corpus"]}
    queries = {e["id"]: e["text"] for e in dataset["queries"]}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["qid"]][e["pid"]] = e["score"]

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs


class T2Retrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="T2Retrieval",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        dataset={
            "path": "mteb/T2Retrieval",
            "revision": "cf778c0ea4168ec5174a34d888d6453e4cde9222",
            # "qrel_revision": "1c83b8d1544e529875e3f6930f3a1fcf749a8e97",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation="""@misc{xie2023t2ranking,
      title={T2Ranking: A large-scale Chinese Benchmark for Passage Ranking}, 
      author={Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
      year={2023},
      eprint={2304.03679},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        prompt={
            "query": "Given a Chinese search query, retrieve web passages that answer the question"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class MMarcoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MMarcoRetrieval",
        description="MMarcoRetrieval",
        reference="https://arxiv.org/abs/2309.07597",
        dataset={
            "path": "mteb/MMarcoRetrieval",
            "revision": "4940a7b26bf53463cfe3435bb8e201963e9c31ae",
            # "qrel_revision": "bae08bb7bddbedb96c7e7db52018a55167b67f89",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation="""@misc{xiao2024cpack,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff and Defu Lian and Jian-Yun Nie},
      year={2024},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class DuRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DuRetrieval",
        description="A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        dataset={
            "path": "mteb/DuRetrieval",
            "revision": "313c81b51311893c8fd09ca432f96b841ed0ebb3",
            # "qrel_revision": "497b7bd1bbb25cb3757ff34d95a8be50a3de2279",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation="""@misc{qiu2022dureaderretrieval,
      title={DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine}, 
      author={Yifu Qiu and Hongyu Li and Yingqi Qu and Ying Chen and Qiaoqiao She and Jing Liu and Hua Wu and Haifeng Wang},
      year={2022},
      eprint={2203.10232},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        prompt={
            "query": "Given a Chinese search query, retrieve web passages that answer the question"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class CovidRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CovidRetrieval",
        description="COVID-19 news articles",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/CovidRetrieval",
            "revision": "9c6dc4b276bb47c3ff725bbc5ffcafd56dded38b",
            # "qrel_revision": "a9f41b7cdf24785531d12417ce0d1157ed4b39ca",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation=None,
        prompt={
            "query": "Given a question on COVID-19, retrieve news articles that answer the question"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class CmedqaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CmedqaRetrieval",
        description="Online medical consultation text. Used the CMedQAv2 as its underlying dataset.",
        reference="https://aclanthology.org/2022.emnlp-main.357.pdf",
        dataset={
            "path": "mteb/CmedqaRetrieval",
            "revision": "c476f85bf03d6642ec66bf54b9a551c88108bbb4",
            # "qrel_revision": "279d737f36c731c8ff6e2b055f31fe02216fa23d",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation=None,
        prompt={
            "query": "Given a Chinese community medical question, retrieve replies that best answer the question"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class EcomRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="EcomRetrieval",
        description="EcomRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/EcomRetrieval",
            "revision": "fa705ce5418e91636b1eaeaf43f34c15aa3f5a8a",
            # "qrel_revision": "39c90699b034ec22ac45b3abf5b0bbb5ffd421f9",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation=None,
        prompt={
            "query": "Given a user query from an e-commerce website, retrieve description sentences of relevant products"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class MedicalRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MedicalRetrieval",
        description="MedicalRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/MedicalRetrieval",
            "revision": "023ae3b2c6b96f583c4ff9b3f9239c93f7885c20",
            # "qrel_revision": "37b8efec53c54c3d9c6af212f6710b62ccdf895c",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation=None,
        prompt={
            "query": "Given a medical question, retrieve user replies that best answer the question"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True


class VideoRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="VideoRetrieval",
        description="VideoRetrieval",
        reference="https://arxiv.org/abs/2203.03367",
        dataset={
            "path": "mteb/VideoRetrieval",
            "revision": "146a9d5e4fd7a9c182b6b92cccb6a3753994305c",
            # "qrel_revision": "faa71382b6a29cf1778d1f436b963e75cb5b927c",
        },
        type="Retrieval",
        category="s2p",
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
        bibtex_citation=None,
        prompt={
            "query": "Given a video search query, retrieve the titles of relevant videos"
        },
    )

    # def load_data(self, **kwargs):
    #     if self.data_loaded:
    #         return
    #
    #     self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
    #         self.metadata_dict["dataset"]["path"],
    #         self.metadata_dict["dataset"]["revision"],
    #         self.metadata_dict["dataset"]["qrel_revision"],
    #         self.metadata_dict["eval_splits"],
    #     )
    #     self.data_loaded = True
