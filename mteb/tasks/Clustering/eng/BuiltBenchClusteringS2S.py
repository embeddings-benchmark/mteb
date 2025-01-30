from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


# NUM_SAMPLES = 2048  # TODO: to be use in dataset_transform method


class BuiltBenchClusteringS2S(AbsTaskClustering):
    superseded_by = None
    metadata = TaskMetadata(
        name="BuiltBenchClusteringS2S",
        description="Clustering of built asset names/titles based on categories identified within industry classification systems such as IFC, Uniclass, etc.",
        reference="https://arxiv.org/abs/2411.12056",
        dataset={
            "path": "mehrzad-shahin/builtbench-clustering-s2s",
            "revision": "1aaeb2ece89ea0a8c64e215c95c4cfaf7e891149",
        },
        type="Clustering",
        category="s2s",
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
        prompt="Identify the category of the built asset entities based on the names or titles",
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
