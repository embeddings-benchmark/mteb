from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class InstructIR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="InstructIR",
        description='A benchmark specifically designed to evaluate the instruction following ability in information retrieval models. Our approach focuses on user-aligned instructions tailored to each query instance, reflecting the diverse characteristics inherent in real-world search scenarios. **NOTE**: scores on this may differ unless you include instruction first, then "[SEP]" and then the query via redefining `combine_query_and_instruction` in your model.',
        reference="https://github.com/kaistAI/InstructIR/tree/main",
        dataset={
            "path": "mteb/InstructIR-mteb",
            "revision": "42c3afabe480643b755a7099dbf0f9ebeedaf6ca",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="robustness_at_10",
        date=("2024-02-05", "2024-02-06"),
        domains=["Web"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{oh2024instructir,
      title={{INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models}}, 
      author={{Hanseok Oh and Hyunji Lee and Seonghyeon Ye and Haebin Shin and Hansol Jang and Changwook Jun and Minjoon Seo}},
      year={{2024}},
      eprint={{2402.14334}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}}
}""",
    )
