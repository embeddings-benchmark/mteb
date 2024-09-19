from __future__ import annotations

import io
import PIL.Image as Image
from mteb.abstasks.Image.AbsTaskImageClustering import AbsTaskImageClustering
from mteb.abstasks.TaskMetadata import TaskMetadata

"""
Classes:
1.MALTESE DOG
2.BLENHEIM SPANIEL
3.BASSET
4.NORWEGIAN ELKHOUND
5.GIANT SCHNAUZER
6.GOLDEN RETRIEVER
7.BRITTANY SPANIEL
8.CLUMBER
9.WELSH SPRINGER SPANIEL
10.GROENENDAEL
11.KELPIE
12.SHETLAND SHEEPDOG
13.DOBERMAN
14.PUG
15.CHOW
"""

class ImageNetDog15Clustering(AbsTaskImageClustering):
    metadata = TaskMetadata(
        name="ImageNetDog15Clustering",
        description="Clustering images from a 15-class dogs-only subset of the dog classes in ImageNet.",
        reference="http://vision.stanford.edu/aditya86/ImageNetDogs/main.html",
        dataset={
            "path": "JamieSJS/imagenet-dog-15",
            "revision": "bfb6ad3b2109d26c9daddf14f98d315daa35ee72",
        },
        type="Clustering",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2009-06-20",
            "2009-06-20"
        ),  # Conference date
        domains=["Web"],
        task_subtypes=["Object recognition"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=""" @INPROCEEDINGS{5206848,
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
  booktitle={2009 IEEE Conference on Computer Vision and Pattern Recognition}, 
  title={ImageNet: A large-scale hierarchical image database}, 
  year={2009},
  volume={},
  number={},
  pages={248-255},
  keywords={Large-scale systems;Image databases;Explosions;Internet;Robustness;Information retrieval;Image retrieval;Multimedia databases;Ontologies;Spine},
  doi={10.1109/CVPR.2009.5206848}}
        """,
        descriptive_stats={
            "n_samples": {"test": 1076, "train":1500},
            #"avg_character_length": {"test": 431.4},
        },
    )

    

