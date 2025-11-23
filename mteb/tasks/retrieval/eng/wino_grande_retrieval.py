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
@article{sakaguchi2021winogrande,
  author = {Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
  journal = {Communications of the ACM},
  number = {9},
  pages = {99--106},
  publisher = {ACM New York, NY, USA},
  title = {Winogrande: An adversarial winograd schema challenge at scale},
  volume = {64},
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
            "query": "Given the following sentence, retrieve an appropriate answer to fill in the missing underscored part."
        },
    )
