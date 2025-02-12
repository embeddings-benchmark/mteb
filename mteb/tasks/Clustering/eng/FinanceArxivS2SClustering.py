from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class FinanceArxivS2SClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="FinanceArxivS2SClustering",
        description="Clustering of titles from arxiv (q-fin).",
        reference="https://arxiv.org/abs/2409.18511v1",
        dataset={
            "path": "FinanceMTEB/FinanceArxiv-s2s",
            "revision": "78f66d3bbea9b1d7f11df84bc55b21b24302dcee",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-09-27", "2024-10-03"),
        domains=["Finance"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        bibtex_citation="""@misc{tang2024needdomainspecificembeddingmodels,
              title={Do We Need Domain-Specific Embedding Models? An Empirical Investigation},
              author={Yixuan Tang and Yi Yang},
              year={2024},
              eprint={2409.18511},
              archivePrefix={arXiv},
              primaryClass={cs.CL},
              url={https://arxiv.org/abs/2409.18511},
        }""",
        descriptive_stats={
            "num_samples": {"test": 32},
            "average_text_length": {"test": 9717.25},
            "average_labels_per_text": {"test": 9717.25},
            "unique_labels": {"test": 9},
            "labels": {
                "test": {
                    "q-fin.PR": {"count": 28920},
                    "q-fin.GN": {"count": 36744},
                    "q-fin.ST": {"count": 47016},
                    "q-fin.CP": {"count": 37888},
                    "q-fin.PM": {"count": 36416},
                    "q-fin.TR": {"count": 33504},
                    "q-fin.EC": {"count": 12320},
                    "q-fin.MF": {"count": 48192},
                    "q-fin.RM": {"count": 29952},
                }
            },
        },
    )
