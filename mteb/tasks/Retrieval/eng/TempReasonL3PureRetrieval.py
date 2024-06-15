from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TempReasonL3Pure(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TempReasonL3Pure",
        description="Reasoning as Retrieval (RAR-b) format: Whether Answers to Queries in Reasoning Tasks can be retrieved as top.",
        reference="https://github.com/DAMO-NLP-SG/TempReason",
        dataset={
            "path": "RAR-b/TempReason-l3-pure",
            "revision": "68fba138e7e63daccecfbdad0a9d2714e56e34ff",
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
        license="CC BY-SA 3.0",
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
@article{tan2023towards,
  title={Towards benchmarking and improving the temporal reasoning capability of large language models},
  author={Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
  journal={arXiv preprint arXiv:2306.08952},
  year={2023}
}
""",
        n_samples=None,
        avg_character_length=None,
    )
