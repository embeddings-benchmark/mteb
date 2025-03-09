from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class OxfordFlowersClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="OxfordFlowersClassification",
        description="Classifying flowers",
        reference="https://huggingface.co/datasets/nelorth/oxford-flowers/viewer/default/train",
        dataset={
            "path": "nelorth/oxford-flowers",
            "revision": "a37b1891609c0376fa81eced756e7863e1bd873b",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2015-12-31",
        ),  # Estimated range for the collection of reviews
        domains=["Reviews"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="found",
        bibtex_citation="""@INPROCEEDINGS{4756141,
  author={Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle={2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing},
  title={Automated Flower Classification over a Large Number of Classes},
  year={2008},
  volume={},
  number={},
  pages={722-729},
  keywords={Shape;Kernel;Distributed computing;Support vector machines;Support vector machine classification;object classification;segmentation},
  doi={10.1109/ICVGIP.2008.47}}""",
        descriptive_stats={
            "n_samples": {"test": 400000},
            "avg_character_length": {"test": 431.4},
        },
    )
