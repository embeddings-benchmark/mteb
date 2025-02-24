from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class WikipediaChemistrySpecialtiesClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikipediaSpecialtiesInChemistryClustering",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-AI/WikipediaMedium5Clustering",
            "revision": "7754d8d296f9f4c3af1c6426fab36304730ccddf",
        },
        type="Clustering",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""
        @article{kasmaee2024chemteb,
        title={ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \& Efficiency on a Specific Domain},
        author={Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
        journal={arXiv preprint arXiv:2412.00532},
        year={2024}
        }
        """,
    )
