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
@article{mirzaee2021spartqa,
  author = {Mirzaee, Roshanak and Faghihi, Hossein Rajaby and Ning, Qiang and Kordjmashidi, Parisa},
  journal = {arXiv preprint arXiv:2104.05832},
  title = {Spartqa:: A textual question answering benchmark for spatial reasoning},
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
