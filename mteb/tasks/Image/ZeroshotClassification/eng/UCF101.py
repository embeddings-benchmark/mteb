from __future__ import annotations

from mteb.abstasks.Image.AbsTaskZeroShotClassification import (
    AbsTaskZeroShotClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class UCF101ZeroShotClassification(AbsTaskZeroShotClassification):
    metadata = TaskMetadata(
        name="UCF101ZeroShot",
        description="""UCF101 is an action recognition data set of realistic
action videos collected from YouTube, having 101 action categories. This
version of the dataset does not contain images but images saved frame by
frame. Train and test splits are generated based on the authors' first
version train/test list.""",
        reference="https://huggingface.co/datasets/flwrlabs/ucf101",
        dataset={
            "path": "flwrlabs/ucf101",
            "revision": "1098eed48f2929443f47c39f3b5c814e16369c11",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2012-01-01",
            "2012-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Scene"],
        task_subtypes=["Activity recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation="""@misc{soomro2012ucf101dataset101human,
      title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
      author={Khurram Soomro and Amir Roshan Zamir and Mubarak Shah},
      year={2012},
      eprint={1212.0402},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1212.0402},
}""",
        descriptive_stats={
            "n_samples": {"test": 697222},
            "avg_character_length": {"test": 0},
        },
    )

    def get_candidate_labels(self) -> list[str]:
        return [
            f"a photo of {name}"
            for name in self.dataset["test"].features[self.label_column_name].names
        ]
