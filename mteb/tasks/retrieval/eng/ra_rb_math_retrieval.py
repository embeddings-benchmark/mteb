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

@article{hendrycks2021measuring,
  author = {Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora, Akul and Basart, Steven and Tang, Eric and Song, Dawn and Steinhardt, Jacob},
  journal = {arXiv preprint arXiv:2103.03874},
  title = {Measuring mathematical problem solving with the math dataset},
  year = {2021},
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
