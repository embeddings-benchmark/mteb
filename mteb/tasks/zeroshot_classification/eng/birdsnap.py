from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BirdsnapZeroShotClassification(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BirdsnapZeroShot",
        description="Classifying bird images from 500 species. ",
        reference="https://openaccess.thecvf.com/content_cvpr_2014/html/Berg_Birdsnap_Large-scale_Fine-grained_2014_CVPR_paper.html",
        dataset={
            "path": "mteb/BirdsnapZeroShot",
            "revision": "05dc46ce92b0d12cc0f8e314e701233718691d90",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2013-01-01",
            "2014-01-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Berg_2014_CVPR,
  author = {Berg, Thomas and Liu, Jiongxin and Woo Lee, Seung and Alexander, Michelle L. and Jacobs, David W. and Belhumeur, Peter N.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  title = {Birdsnap: Large-scale Fine-grained Visual Categorization of Birds},
  year = {2014},
}
""",
    )
