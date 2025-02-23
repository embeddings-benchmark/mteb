from __future__ import annotations

from mteb.abstasks.Image import AbsTaskAny2AnyRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VQA2IT2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VQA2IT2TRetrieval",
        description="Retrieve the correct answer for a question about an image.",
        reference="https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html",
        dataset={
            "path": "JamieSJS/vqa-2",
            "revision": "69882b6ba0b443dd62e633e546725b0f13b7e3aa",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-07-01", "2017-07-01"),
        domains=["Web"],
        task_subtypes=["image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@InProceedings{Goyal_2017_CVPR,
author = {Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
title = {Making the v in VQA Matter: Elevating the Role of image Understanding in Visual Question Answering},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
""",
    )
