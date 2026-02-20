from pathlib import Path

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.zeroshot_classification import (
    AbsTaskZeroShotClassification,
)


class Imagenet1kZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="Imagenet1kZeroShot",
        description="ImageNet, a large-scale ontology of images built upon the backbone of the WordNet structure.",
        reference="https://ieeexplore.ieee.org/document/5206848",
        dataset={
            "path": "mteb/wds_imagenet1k",
            "revision": "b54c9af96641d8a2cba5d1144629c724de94ef76",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2010-01-01",
            "2012-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["image", "text"],
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
    input_column_name: str = "jpg"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = Path(__file__).parent / "templates" / "Imagenet1k_labels.txt"
        with path.open() as f:
            labels = f.readlines()

        return [f"a photo of {c}." for c in labels]
