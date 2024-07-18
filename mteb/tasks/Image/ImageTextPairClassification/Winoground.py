from __future__ import annotations

from datasets import Features, Image, Sequence, Value

from mteb.abstasks import AbsTaskImageTextPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Winoground(AbsTaskImageTextPairClassification):

    images_column_names = ["image_0","image_1"]
    texts_column_names = ["caption_0","caption_1"]
    
    metadata = TaskMetadata(
        name="Winoground",
        description="Compositionality Evaluation of images to their captions.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper",
        dataset={
            "path": "facebook/winoground",
            "revision": "521ec2ba6f9a5d7380f7cca5a7b44aea5c1d677c",
            "trust_remote_code": True,
        },
        type="ImageTextPairClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2022-01-01",
            "2022-04-07",
        ),  # Estimated range for the collection of data
        form=["written"],
        domains=["Social"],  # Getty Images. Could be constructed?
        task_subtypes=["Caption Pairing"],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{thrush2022winogroundprobingvisionlanguage,
            title={Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality}, 
            author={Tristan Thrush and Ryan Jiang and Max Bartolo and Amanpreet Singh and Adina Williams and Douwe Kiela and Candace Ross},
            year={2022},
            eprint={2204.03162},
            archivePrefix={arXiv},
            primaryClass={cs.CV},
            url={https://arxiv.org/abs/2204.03162}, 
        }""",
        n_samples={"test": 400},
        avg_character_length={"test": 431.4},
    )

    # def dataset_transform(self):
    #     def group_data(example):
    #         example["images"] = [example["image_0"], example["image_1"]]
    #         example["captions"] = [example["caption_0"], example["caption_1"], "It's whatever"]
    #         return example

    #     orig_col_names = self.dataset["test"].column_names
    #     # self.dataset["test"] = self.dataset["test"].select([0, 1, 2, 3])
    #     self.dataset["test"] = self.dataset["test"].map(group_data)

    #     self.dataset["test"] = self.dataset["test"].remove_columns(orig_col_names)
    #     features = Features(
    #         {"images": Sequence(Image()), "captions": Sequence(Value("string"))}
    #     )
    #     self.dataset["test"] = self.dataset["test"].cast(features)

    #     return
