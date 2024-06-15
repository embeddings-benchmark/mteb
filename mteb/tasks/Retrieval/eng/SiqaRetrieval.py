from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SIQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SIQA",
        description="Reasoning as Retrieval (RAR-b) format: Whether Answers to Queries in Reasoning Tasks can be retrieved as top.",
        reference="https://leaderboard.allenai.org/socialiqa/submissions/get-started",
        dataset={
            "path": "RAR-b/siqa",
            "revision": "4ed8415e9dc24060deefc84be59e2db0aacbadcc",
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
        license="CC BY",
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
@article{sap2019socialiqa,
  title={Socialiqa: Commonsense reasoning about social interactions},
  author={Sap, Maarten and Rashkin, Hannah and Chen, Derek and LeBras, Ronan and Choi, Yejin},
  journal={arXiv preprint arXiv:1904.09728},
  year={2019}
}
""",
        n_samples=None,
        avg_character_length=None,
    )
