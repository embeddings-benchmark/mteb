import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBQMSumRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBQMSumRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "qmsum",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("qmsum subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1950-01-01", "2021-12-31"),
        domains=["Spoken", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{zhong-etal-2021-qmsum,
  address = {Online},
  author = {Zhong, Ming  and
Yin, Da  and
Yu, Tao  and
Zaidi, Ahmad  and
Mutuma, Mutethia  and
Jha, Rahul  and
Awadallah, Ahmed Hassan  and
Celikyilmaz, Asli  and
Liu, Yang  and
Qiu, Xipeng  and
Radev, Dragomir},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.472},
  editor = {Toutanova, Kristina  and
Rumshisky, Anna  and
Zettlemoyer, Luke  and
Hakkani-Tur, Dilek  and
Beltagy, Iz  and
Bethard, Steven  and
Cotterell, Ryan  and
Chakraborty, Tanmoy  and
Zhou, Yichao},
  month = jun,
  pages = {5905--5921},
  publisher = {Association for Computational Linguistics},
  title = {{QMS}um: A New Benchmark for Query-based Multi-domain Meeting Summarization},
  url = {https://aclanthology.org/2021.naacl-main.472},
  year = {2021},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata.dataset)[
            "queries"
        ]  # dict_keys(['qid', 'text'])
        queries_dict = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata.dataset)[
            "corpus"
        ]  # dict_keys(['doc_id', 'text'])
        corpus_dict = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata.dataset)[
            "qrels"
        ]  # dict_keys(['qid', 'doc_id'])
        relevant_docs = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        corpus_dataset = Dataset.from_list(
            [
                {
                    "id": k,
                    "text": v.get("text", "") if isinstance(v, dict) else v,
                    "title": v.get("title", "") if isinstance(v, dict) else "",
                }
                for k, v in corpus_dict.items()
            ]
        )
        queries_dataset = Dataset.from_list(
            [{"id": k, "text": v} for k, v in queries_dict.items()]
        )

        self.dataset = {
            "default": {
                self._EVAL_SPLIT: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }

        self.data_loaded = True
