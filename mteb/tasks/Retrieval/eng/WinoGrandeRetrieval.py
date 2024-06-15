from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class WinoGrande(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WinoGrande",
        description="Reasoning as Retrieval (RAR-b) format: Whether Answers to Queries in Reasoning Tasks can be retrieved as top.",
        reference="https://winogrande.allenai.org/",
        dataset={
            "path": "RAR-b/winogrande",
            "revision": "f74c094f321077cf909ddfb8bccc1b5912a4ac28",
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
}""",
        n_samples=None,
        avg_character_length=None,
    )
