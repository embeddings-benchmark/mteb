from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class BIRCODorisMaeReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="BIRCO-DorisMae",
        description=(
            "Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries "
            "that are complex research questions from computer scientists. Each query has a candidate pool of "
            "approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant)."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2s",  # MTEB standard category
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-DorisMae-Test",
            "revision": "2429b297e5d73663d20bc885064e5599bb29d74a",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic"],  # Valid combination
        task_subtypes=["Scientific Reranking"],  # MTEB-approved subtype
        license="https://creativecommons.org/licenses/by/4.0/",  # Full URL
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Identify scientific abstracts that fulfill research requirements",
        bibtex_citation="""@misc{wang2024bircobenchmarkinformationretrieval,
            title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives}, 
            author={Xiaoyue Wang and Jianyou Wang and Weili Cao and Kaicheng Wang and Ramamohan Paturi and Leon Bergen},
            year={2024},
            eprint={2402.14151},
            archivePrefix={arXiv},
            primaryClass={cs.IR},
            url={https://arxiv.org/abs/2402.14151}, 
        }""",
    )
