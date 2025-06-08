from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Quail(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quail",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on Quail.",
        reference="https://text-machine.cs.uml.edu/lab2/projects/quail/",
        dataset={
            "path": "RAR-b/quail",
            "revision": "1851bc536f8bdab29e03e29191c4586b1d8d7c5a",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{rogers2020getting,
  author = {Rogers, Anna and Kovaleva, Olga and Downey, Matthew and Rumshisky, Anna},
  booktitle = {Proceedings of the AAAI conference on artificial intelligence},
  number = {05},
  pages = {8722--8731},
  title = {Getting closer to AI complete question answering: A set of prerequisite real tasks},
  volume = {34},
  year = {2020},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following context and question, retrieve the correct answer."
        },
    )
