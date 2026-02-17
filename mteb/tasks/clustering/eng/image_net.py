from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class ImageNetDog15Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="ImageNetDog15Clustering",
        description="Clustering images from a 15-class dogs-only subset of the dog classes in ImageNet.",
        reference="http://vision.stanford.edu/aditya86/ImageNetDogs/main.html",
        dataset={
            "path": "mteb/imagenet-dog-15",
            "revision": "16b4f3ff0eba385e23756afc3fdefc460a5c3444",
        },
        type="ImageClustering",
        category="i2c",
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
@inproceedings{deng2009imagenet,
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
    )
    input_column_name = "image"
    label_column_name = "label"


class ImageNet10Clustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="ImageNet10Clustering",
        description="Clustering images from an 10-class subset of ImageNet which are generally easy to distinguish.",
        reference="https://www.kaggle.com/datasets/liusha249/imagenet10",
        dataset={
            "path": "mteb/imagenet-10",
            "revision": "6ee8458695d029cc225d046deb33342dfca9d4ce",
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
@inproceedings{deng2009imagenet,
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
    )
    input_column_name = "image"
    label_column_name = "label"
