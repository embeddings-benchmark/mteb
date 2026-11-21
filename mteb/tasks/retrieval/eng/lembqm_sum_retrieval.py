from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBQMSumRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBQMSumRetrieval",
        dataset={
            "path": "mteb/LEMBQMSumRetrieval",
            "revision": "e40544d73a518df7f1efd876aced9eae22459a8c",
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
