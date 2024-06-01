from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class BIRCODorisMaeReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="BIRCODorisMaeReranking",
        description="BIRCO:A Benchmark of Information Retrieval Tasks with Complex Objectives. 60 queries that are complex research questions from computer scientists. The query communicates specific requirements from research papers. Candidate pools have approximately 110 documents.",
        reference="https://github.com/DPWXY/BIRCO/tree/main",
        dataset={
            "path": "bpHigh/BIRCO_Doris_Mae_Without_Task_Awareness",
            "revision": "451efcc2986eaa24db8a71a19c68be44352e083c",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-02-21", "2020-04-03"),
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Scientific Reranking"],
        license="CC-BY-NC 4.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{wang2024birco,
          title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives}, 
          author={Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
          year={2024},
          eprint={2402.14151},
          archivePrefix={arXiv},
          primaryClass={cs.IR}
    }""",
        n_samples={"test": 60},
        avg_character_length={"test": 995.55},
    )
