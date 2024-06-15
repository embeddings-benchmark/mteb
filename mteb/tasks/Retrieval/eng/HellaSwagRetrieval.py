from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HellaSwag(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HellaSwag",
        description="Reasoning as Retrieval (RAR-b) format: Whether Answers to Queries in Reasoning Tasks can be retrieved as top.",
        reference="https://rowanzellers.com/hellaswag/",
        dataset={
            "path": "RAR-b/hellaswag",
            "revision": "a5c990205e017d10761197ccab3000936689c3ae",
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
@article{zellers2019hellaswag,
  title={Hellaswag: Can a machine really finish your sentence?},
  author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
  journal={arXiv preprint arXiv:1905.07830},
  year={2019}
}
""",
        n_samples=None,
        avg_character_length=None,
    )
