from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WinoGrande(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WinoGrande",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on winogrande.",
        reference="https://winogrande.allenai.org/",
        dataset={
            "path": "mteb/WinoGrande",
            "revision": "4dec9c5666e9f84702ac614363db6d96a68bc6de",
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
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{sakaguchi_winogrande_2021,
  author = {Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
  title = {WinoGrande: an adversarial winograd schema challenge at scale},
  year = {2021},
  issue_date = {September 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {64},
  number = {9},
  issn = {0001-0782},
  url = {https://doi.org/10.1145/3474381},
  doi = {10.1145/3474381},
  journal = {Communications of the ACM},
  month = aug,
  pages = {99–106},
  numpages = {8},
}

@misc{xiao_rar-b_2024,
  archiveprefix = {arXiv},
  author = {Chenghao Xiao and G Thomas Hudson and Noura Al Moubayed},
  eprint = {2404.06347},
  primaryclass = {cs.CL},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  url = {https://arxiv.org/abs/2404.06347},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following sentence, retrieve an appropriate answer to fill in the missing underscored part."
        },
    )
