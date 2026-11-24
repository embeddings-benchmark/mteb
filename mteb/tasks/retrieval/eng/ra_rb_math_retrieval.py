from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class RARbMath(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RARbMath",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.",
        reference="https://arxiv.org/abs/2404.06347",
        dataset={
            "path": "mteb/RARbMath",
            "revision": "cc879a9719a10f4a80677ac26798cfd701cf94b7",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{cobbe2021training,
  author = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and others},
  journal = {arXiv preprint arXiv:2110.14168},
  title = {Training verifiers to solve math word problems},
  year = {2021},
}

@inproceedings{hendrycks2021math,
  archiveprefix = {arXiv},
  eprint = {2103.03874},
  title = {Measuring Mathematical Problem Solving With the {MATH} Dataset},
  author = {Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year = {2021},
  url = {https://openreview.net/forum?id=7Bywt2mQsCe},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}

@inproceedings{yu2023metamath,
  archiveprefix = {arXiv},
  author = {Longhui Yu and Weisen Jiang and Han Shi and Jincheng YU and Zhengying Liu and Yu Zhang and James Kwok and Zhenguo Li and Adrian Weller and Weiyang Liu},
  booktitle = {The Twelfth International Conference on Learning Representations},
  eprint = {2309.12284},
  title = {MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models},
  url = {https://openreview.net/forum?id=N8N0hgNDRt},
  year = {2024},
}
""",
        prompt={"query": "Retrieve the answer for the following math problem."},
    )
