from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SpartQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpartQA",
        description="Reasoning as Retrieval (RAR-b) format: Whether Answers to Queries in Reasoning Tasks can be retrieved as top.",
        reference="https://github.com/HLR/SpartQA_generation",
        dataset={
            "path": "RAR-b/spartqa",
            "revision": "9ab3ca3ccdd0d43f9cd6d346a363935d127f4f45",
        },
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=[],
        license="MIT",
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
@article{mirzaee2021spartqa,
  title={Spartqa:: A textual question answering benchmark for spatial reasoning},
  author={Mirzaee, Roshanak and Faghihi, Hossein Rajaby and Ning, Qiang and Kordjmashidi, Parisa},
  journal={arXiv preprint arXiv:2104.05832},
  year={2021}
}
""",
        n_samples={"test": 0},
        avg_character_length={"test": 0.0},
    )
