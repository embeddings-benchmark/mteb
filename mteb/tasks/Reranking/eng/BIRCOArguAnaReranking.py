from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BIRCOArguAnaReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BIRCO-ArguAna",
        description=(
            "Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries "
            "and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument "
            "that directly refutes the query’s stance."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-ArguAna-Test",
            "revision": "0229bee36500c92028874b55b5a1d08e5ee37bb9",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Written"],  # there is no 'Debate' domain
        task_subtypes=["Reasoning as Retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Given a one-paragraph argument, retrieve the passage that contains the counter-argument which directly refutes the query's stance.",
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
