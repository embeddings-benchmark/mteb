from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class Quail(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Quail",
        description="Reasoning as Retrieval (RAR-b) format: Whether Answers to Queries in Reasoning Tasks can be retrieved as top.",
        reference="https://text-machine.cs.uml.edu/lab2/projects/quail/",
        dataset={
            "path": "RAR-b/quail",
            "revision": "1851bc536f8bdab29e03e29191c4586b1d8d7c5a",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@article{xiao2024rar,
  title={RAR-b: Reasoning as Retrieval Benchmark},
  author={Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2404.06347},
  year={2024}
}
@inproceedings{rogers2020getting,
  title={Getting closer to AI complete question answering: A set of prerequisite real tasks},
  author={Rogers, Anna and Kovaleva, Olga and Downey, Matthew and Rumshisky, Anna},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={05},
  pages={8722--8731},
  year={2020}
}
""",
        n_samples=None,
        avg_character_length=None,
    )
