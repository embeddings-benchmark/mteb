from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TempReasonL3Fact(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TempReasonL3Fact",
        description="Measuring the ability to retrieve the groundtruth answers to reasoning task queries on TempReason l3-fact.",
        reference="https://github.com/DAMO-NLP-SG/TempReason",
        dataset={
            "path": "mteb/TempReasonL3Fact",
            "revision": "ce6502fe436f00f05a73fe24ed0317c57fe4d6b0",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="cc-by-sa-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{tan2023towards,
  author = {Tan, Qingyu and Ng, Hwee Tou and Bing, Lidong},
  journal = {arXiv preprint arXiv:2306.08952},
  title = {Towards benchmarking and improving the temporal reasoning capability of large language models},
  year = {2023},
}

@article{xiao2024rar,
  author = {Xiao, Chenghao and Hudson, G Thomas and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2404.06347},
  title = {RAR-b: Reasoning as Retrieval Benchmark},
  year = {2024},
}
""",
        prompt={
            "query": "Given the following question and facts, retrieve the correct answer."
        },
    )
