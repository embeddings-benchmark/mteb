from __future__ import annotations

from mteb.abstasks import AbsTaskImageTextPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Winoground(AbsTaskImageTextPairClassification):
    metadata = TaskMetadata(
        name="Winoground",
        description="Compositionality Evaluation of images to their captions.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper",
        dataset={
            "path": "facebook/winoground",
            "revision": "5a77092c28e51558c5586e9c5eb71a7e17a5e43f",
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
    #     self.dataset['test'] = self.dataset['test'].select([0,1,2,3])
    #     return
