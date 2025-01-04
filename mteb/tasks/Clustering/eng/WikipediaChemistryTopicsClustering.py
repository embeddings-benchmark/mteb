from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class WikipediaChemistryTopicsClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="WikipediaChemistryTopicsClustering",
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        dataset={
            "path": "BASF-We-Create-Chemistry/WikipediaEasy10Clustering",
            "revision": "0a0886b06acbfc735bca6a71b21ce1e5cb92a37b",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=None,
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators=None,
        dialect=[],
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
