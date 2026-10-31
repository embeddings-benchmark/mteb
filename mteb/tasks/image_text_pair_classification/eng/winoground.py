from mteb.abstasks.image.image_text_pair_classification import (
    AbsTaskImageTextPairClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata


class Winoground(AbsTaskImageTextPairClassification):
    images_column_names = ("image_0", "image_1")
    texts_column_names = ("caption_0", "caption_1")

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
        bibtex_citation=r"""
@inproceedings{thrush2022winogroundprobingvisionlanguage,
  author = {Thrush, Tristan and Jiang, Ryan and Bartolo, Max and Singh, Amanpreet and Williams, Adina and Kiela, Douwe and Ross, Candace},
  booktitle = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  doi = {10.1109/cvpr52688.2022.00517},
  month = {June},
  pages = {5228–5238},
  publisher = {IEEE},
  title = {Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality},
  url = {https://doi.org/10.1109/cvpr52688.2022.00517},
  year = {2022},
}
""",
    )
