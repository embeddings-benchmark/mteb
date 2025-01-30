from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


# NUM_SAMPLES = 2048  # TODO: to be use in dataset_transform method


class BuiltBenchClusteringP2P(AbsTaskClustering):
    superseded_by = None
    metadata = TaskMetadata(
        name="BuiltBenchClusteringP2P",
        description="Clustering of built asset item descriptions based on categories identified within industry classification systems such as IFC, Uniclass, etc.",
        reference="https://arxiv.org/abs/2411.12056",
        dataset={
            "path": "mehrzad-shahin/builtbench-clustering-p2p",
            "revision": "919bb71053e9de62a68998161ce4f0cee8f786fb", 
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=None,
        domains=None,
        task_subtypes=None,
        license="cc-by-nd-4.0",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@article{shahinmoghadam2024benchmarking,
    title={Benchmarking pre-trained text embedding models in aligning built asset information},
    author={Shahinmoghadam, Mehrzad and Motamedi, Ali},
    journal={arXiv preprint arXiv:2411.12056},
    year={2024}
}""",
        prompt="Identify the category of the built asset entities based on the entity description",
    )


#     def dataset_transform(self):
#         for split in self.metadata.eval_splits:
#             check_label_distribution(self.dataset[split])
#         self.dataset = self.stratified_subsampling(
#             self.dataset,
#             self.seed,
#             self.metadata.eval_splits,
#             label="labels",
#             n_samples=NUM_SAMPLES,
#         )
