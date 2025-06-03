from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RARbMath(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RARbMath",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on RAR-b math-pooled dataset.",
        reference="https://arxiv.org/abs/2404.06347",
        dataset={
            "path": "RAR-b/math-pooled",
            "revision": "2393603c0221ff52f448d12dd75f0856103c6cca",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
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

@article{yu2023metamath,
  author = {Yu, Longhui and Jiang, Weisen and Shi, Han and Yu, Jincheng and Liu, Zhengying and Zhang, Yu and Kwok, James T and Li, Zhenguo and Weller, Adrian and Liu, Weiyang},
  journal = {arXiv preprint arXiv:2309.12284},
  title = {Metamath: Bootstrap your own mathematical questions for large language models},
  year = {2023},
}
""",
        prompt={"query": "Retrieve the answer for the following math problem."},
    )
