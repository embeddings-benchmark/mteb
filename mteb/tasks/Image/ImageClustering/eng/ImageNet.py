from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClustering import AbsTaskImageClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class ImageNetDog15Clustering(AbsTaskImageClustering):
    metadata = TaskMetadata(
        name="ImageNetDog15Clustering",
        description="Clustering images from a 15-class dogs-only subset of the dog classes in ImageNet.",
        reference="http://vision.stanford.edu/aditya86/ImageNetDogs/main.html",
        dataset={
            "path": "JamieSJS/imagenet-dog-15",
            "revision": "bfb6ad3b2109d26c9daddf14f98d315daa35ee72",
        },
        type="ImageClustering",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=("2009-06-20", "2009-06-20"),  # Conference date
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{5206848,
  author = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
  booktitle = {2009 IEEE Conference on Computer Vision and Pattern Recognition},
  doi = {10.1109/CVPR.2009.5206848},
  keywords = {Large-scale systems;Image databases;Explosions;Internet;Robustness;Information retrieval;Image retrieval;Multimedia databases;Ontologies;Spine},
  number = {},
  pages = {248-255},
  title = {ImageNet: A large-scale hierarchical image database},
  volume = {},
  year = {2009},
}
""",
        descriptive_stats={
            "n_samples": {"test": 1076, "train": 1500},
            # "avg_character_length": {"test": 431.4},
        },
    )


class ImageNet10Clustering(AbsTaskImageClustering):
    metadata = TaskMetadata(
        name="ImageNet10Clustering",
        description="Clustering images from an 10-class subset of ImageNet which are generally easy to distinguish.",
        reference="https://www.kaggle.com/datasets/liusha249/imagenet10",
        dataset={
            "path": "JamieSJS/imagenet-10",
            "revision": "88f8a6d47c257895094c5ad81e67ba751771fc99",
        },
        type="ImageClustering",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="nmi",
        date=("2009-06-20", "2009-06-20"),  # Conference date
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{5206848,
  author = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
  booktitle = {2009 IEEE Conference on Computer Vision and Pattern Recognition},
  doi = {10.1109/CVPR.2009.5206848},
  keywords = {Large-scale systems;Image databases;Explosions;Internet;Robustness;Information retrieval;Image retrieval;Multimedia databases;Ontologies;Spine},
  number = {},
  pages = {248-255},
  title = {ImageNet: A large-scale hierarchical image database},
  volume = {},
  year = {2009},
}
""",
        descriptive_stats={
            "n_samples": {"test": 13000},
            # "avg_character_length": {"test": 431.4},
        },
    )
