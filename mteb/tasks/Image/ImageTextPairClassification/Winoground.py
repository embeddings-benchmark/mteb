from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageTextPairClassification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class Winoground(AbsTaskImageTextPairClassification):
    images_column_names = ["image_0", "image_1"]
    texts_column_names = ["caption_0", "caption_1"]

    metadata = TaskMetadata(
        name="Winoground",
        description="Compositionality Evaluation of images to their captions.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper",
        dataset={
            "path": "facebook/winoground",
            "revision": "b400e173549071916ad1b3d449293bc8d8b4b763",
        },
        type="Compositionality",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2022-01-01",
            "2022-04-07",
        ),  # Estimated range for the collection of data
        domains=["Social"],  # Getty Images. Could be constructed?
        task_subtypes=["Caption Pairing"],
        license="https://huggingface.co/datasets/facebook/winoground/blob/main/license_agreement.txt",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation="""@misc{thrush2022winogroundprobingvisionlanguage,
            title={Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality},
            author={Tristan Thrush and Ryan Jiang and Max Bartolo and Amanpreet Singh and Adina Williams and Douwe Kiela and Candace Ross},
            year={2022},
            eprint={2204.03162},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2204.03162},
        }""",
        descriptive_stats={
            "n_samples": {"test": 400},
            "avg_character_length": {"test": 431.4},
        },
    )
