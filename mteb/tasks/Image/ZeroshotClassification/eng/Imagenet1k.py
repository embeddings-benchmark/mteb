from __future__ import annotations

import itertools
import os

from mteb.abstasks import AbsTaskZeroshotClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Imagenet1kClassification(AbsTaskZeroshotClassification):
    metadata = TaskMetadata(
        name="Imagenet1kZeroShot",
        description="ImageNet, a large-scale ontology of images built upon the backbone of the WordNet structure.",
        reference="https://ieeexplore.ieee.org/document/5206848",
        dataset={
            "path": "clip-benchmark/wds_imagenet1k",
            "revision": "b24c7a5a3ef12df09089055d1795e2ce7c7e7397",
        },
        type="Classification",
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
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@article{deng2009imagenet,
        title={ImageNet: A large-scale hierarchical image database},
        author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
        journal={2009 IEEE Conference on Computer Vision and Pattern Recognition},
        pages={248--255},
        year={2009},
        organization={Ieee}
        }""",
        descriptive_stats={
            "n_samples": {"test": 37200},
            "avg_character_length": {"test": 0},
        },
    )
    image_column_name: str = "jpg"
    label_column_name: str = "cls"

    def get_candidate_labels(self) -> list[str]:
        path = os.path.dirname(__file__)
        with open(os.path.join(path, "templates/Imagenet1k_labels.txt")) as f:
            labels = f.readlines()

        print(len(labels))

        prompts = [
            [f"a bad photo of a {c}." for c in labels],
            [f"a photo of many {c}." for c in labels],
            [f"a photo of a sculpture of a {c}." for c in labels],
            [f"a photo of the hard to see {c}." for c in labels],
            [f"a low resolution photo of the {c}." for c in labels],
            [f"a rendering of a {c}." for c in labels],
            [f"graffiti of a {c}." for c in labels],
            [f"a bad photo of the {c}." for c in labels],
            [f"a cropped photo of the {c}." for c in labels],
            [f"a tattoo of a {c}." for c in labels],
            [f"the embroidered {c}." for c in labels],
            [f"a photo of a hard to see {c}." for c in labels],
            [f"a bright photo of a {c}." for c in labels],
            [f"a photo of a clean {c}." for c in labels],
            [f"a photo of a dirty {c}." for c in labels],
            [f"a dark photo of the {c}." for c in labels],
            [f"a drawing of a {c}." for c in labels],
            [f"a photo of my {c}." for c in labels],
            [f"the plastic {c}." for c in labels],
            [f"a photo of the cool {c}." for c in labels],
            [f"a close-up photo of a {c}." for c in labels],
            [f"a black and white photo of the {c}." for c in labels],
            [f"a painting of the {c}." for c in labels],
            [f"a painting of a {c}." for c in labels],
            [f"a pixelated photo of the {c}." for c in labels],
            [f"a sculpture of the {c}." for c in labels],
            [f"a bright photo of the {c}." for c in labels],
            [f"a cropped photo of a {c}." for c in labels],
            [f"a plastic {c}." for c in labels],
            [f"a photo of the dirty {c}." for c in labels],
            [f"a jpeg corrupted photo of a {c}." for c in labels],
            [f"a blurry photo of the {c}." for c in labels],
            [f"a photo of the {c}." for c in labels],
            [f"a good photo of the {c}." for c in labels],
            [f"a rendering of the {c}." for c in labels],
            [f"a {c} in a video game." for c in labels],
            [f"a photo of one {c}." for c in labels],
            [f"a doodle of a {c}." for c in labels],
            [f"a close-up photo of the {c}." for c in labels],
            [f"a photo of a {c}." for c in labels],
            [f"the origami {c}." for c in labels],
            [f"the {c} in a video game." for c in labels],
            [f"a sketch of a {c}." for c in labels],
            [f"a doodle of the {c}." for c in labels],
            [f"a origami {c}." for c in labels],
            [f"a low resolution photo of a {c}." for c in labels],
            [f"the toy {c}." for c in labels],
            [f"a rendition of the {c}." for c in labels],
            [f"a photo of the clean {c}." for c in labels],
            [f"a photo of a large {c}." for c in labels],
            [f"a rendition of a {c}." for c in labels],
            [f"a photo of a nice {c}." for c in labels],
            [f"a photo of a weird {c}." for c in labels],
            [f"a blurry photo of a {c}." for c in labels],
            [f"a cartoon {c}." for c in labels],
            [f"art of a {c}." for c in labels],
            [f"a sketch of the {c}." for c in labels],
            [f"a embroidered {c}." for c in labels],
            [f"a pixelated photo of a {c}." for c in labels],
            [f"itap of the {c}." for c in labels],
            [f"a jpeg corrupted photo of the {c}." for c in labels],
            [f"a good photo of a {c}." for c in labels],
            [f"a plushie {c}." for c in labels],
            [f"a photo of the nice {c}." for c in labels],
            [f"a photo of the small {c}." for c in labels],
            [f"a photo of the weird {c}." for c in labels],
            [f"the cartoon {c}." for c in labels],
            [f"art of the {c}." for c in labels],
            [f"a drawing of the {c}." for c in labels],
            [f"a photo of the large {c}." for c in labels],
            [f"a black and white photo of a {c}." for c in labels],
            [f"the plushie {c}." for c in labels],
            [f"a dark photo of a {c}." for c in labels],
            [f"itap of a {c}." for c in labels],
            [f"graffiti of the {c}." for c in labels],
            [f"a toy {c}." for c in labels],
            [f"itap of my {c}." for c in labels],
            [f"a photo of a cool {c}." for c in labels],
            [f"a photo of a small {c}." for c in labels],
            [f"a tattoo of the {c}." for c in labels],
        ]
        return itertools.chain(*prompts)
