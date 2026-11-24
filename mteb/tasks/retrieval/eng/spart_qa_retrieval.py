from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SpartQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpartQA",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on SpartQA.",
        reference="https://github.com/HLR/SpartQA_generation",
        dataset={
            "path": "mteb/SpartQA",
            "revision": "1c858df377e57725a014a1b7321ebd79d62016b6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{mirzaee2021spartqa,
  address = {Online},
  author = {Mirzaee, Roshanak and Rajaby Faghihi, Hossein and Ning, Qiang and Kordjamshidi, Parisa},
  booktitle = {Proceedings of the 2021 Conference of the {N}orth {A}merican Chapter of the {A}ssociation for {C}omputational {L}inguistics: Human Language Technologies},
  doi = {10.18653/v1/2021.naacl-main.364},
  editor = {Toutanova, Kristina and Rumshisky, Anna and Zettlemoyer, Luke and Hakkani-Tur, Dilek and Beltagy, Iz and Bethard, Steven and Cotterell, Ryan and Chakraborty, Tanmoy and Zhou, Yichao},
  month = {jun},
  pages = {4582--4598},
  publisher = {Association for Computational Linguistics},
  title = {{SPARTQA}: A Textual Question Answering Benchmark for Spatial Reasoning},
  url = {https://aclanthology.org/2021.naacl-main.364/},
  year = {2021},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following spatial reasoning question, retrieve the right answer."
        },
    )
