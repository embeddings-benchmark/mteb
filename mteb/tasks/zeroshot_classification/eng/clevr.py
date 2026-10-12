from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CLEVR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CLEVRZeroShot",
        description="CLEVR closest object distance identification task.",
        reference="https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html",
        dataset={
            "path": "mteb/CLEVRZeroShot",
            "revision": "7a8fadc65644f6e245fd779cb55e41986081b608",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2016-12-20"),
        domains=["Constructed"],
        task_subtypes=["Object recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Johnson_2017_CVPR,
  author = {Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and Fei-Fei, Li and Lawrence Zitnick, C. and Girshick, Ross},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  title = {CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  year = {2017},
}
""",
    )


class CLEVRCount(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CLEVRCountZeroShot",
        description="CLEVR count objects task.",
        reference="https://openaccess.thecvf.com/content_cvpr_2017/html/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.html",
        dataset={
            "path": "mteb/CLEVRCountZeroShot",
            "revision": "04ddfa8dc86ecec1fba9541193cce29e2d176011",
        },
        type="ZeroShotClassification",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2016-12-20"),
        domains=["Constructed"],
        task_subtypes=["Object recognition"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Johnson_2017_CVPR,
  author = {Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and Fei-Fei, Li and Lawrence Zitnick, C. and Girshick, Ross},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  title = {CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  year = {2017},
}
""",
    )
