from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class AlphaNLI(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlphaNLI",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on AlphaNLI.",
        reference="https://leaderboard.allenai.org/anli/submissions/get-started",
        dataset={
            "path": "RAR-b/alphanli",
            "revision": "303f40ef3d50918d3dc43577d33f2f7344ad72c1",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{bhagavatula2019abductive,
  author = {Bhagavatula, Chandra and Bras, Ronan Le and Malaviya, Chaitanya and Sakaguchi, Keisuke and Holtzman, Ari and Rashkin, Hannah and Downey, Doug and Yih, Scott Wen-tau and Choi, Yejin},
  journal = {arXiv preprint arXiv:1908.05739},
  title = {Abductive commonsense reasoning},
  year = {2019},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following start and end of a story, retrieve a possible reason that leads to the end."
        },
    )
