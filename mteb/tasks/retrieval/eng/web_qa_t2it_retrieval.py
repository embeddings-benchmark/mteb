from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WebQAT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WebQAT2ITRetrieval",
        description="Retrieve sources of information based on questions.",
        reference="https://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html",
        dataset={
            "path": "mteb/mbeir_webqa_task2",
            "revision": "8a9243d504a2039b3467abdef2208a127a5c0737",
        },
        type="Any2AnyRetrieval",
        category="t2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{chang2022webqa,
  author = {Chang, Yingshan and Narang, Mridu and Suzuki, Hisami and Cao, Guihong and Gao, Jianfeng and Bisk, Yonatan},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {16495--16504},
  title = {Webqa: Multihop and multimodal qa},
  year = {2022},
}
""",
        prompt={"query": "Find a Wikipedia image that answers this question."},
    )
