from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ChemHotpotQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ChemHotpotQARetrieval",
        dataset={
            "path": "BASF-AI/ChemHotpotQARetrieval",
            "revision": "1840e8a5ac6ec752bbdd97d543ead0189bc7c25b",
        },
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["train", "dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Chemistry"],
        task_subtypes=None,
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation=None,
        bibtex_citation="""
        @article{kasmaee2024chemteb,
        title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
        author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
        journal={arXiv preprint arXiv:2412.00532},
        year={2024}
        }
        """,
    )
