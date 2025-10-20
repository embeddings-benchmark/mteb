from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VQA2IT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VQA2IT2TRetrieval",
        description="Retrieve the correct answer for a question about an image.",
        reference="https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html",
        dataset={
            "path": "JamieSJS/vqa-2",
            "revision": "69882b6ba0b443dd62e633e546725b0f13b7e3aa",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-07-01", "2017-07-01"),
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Goyal_2017_CVPR,
  author = {Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  title = {Making the v in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering},
  year = {2017},
}
""",
    )
