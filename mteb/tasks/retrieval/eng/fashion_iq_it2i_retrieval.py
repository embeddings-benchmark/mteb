from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FashionIQIT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FashionIQIT2IRetrieval",
        description="Retrieve clothes based on descriptions.",
        reference="https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html",
        dataset={
            "path": "mteb/mbeir_fashioniq_task7",
            "revision": "469fba95d895f129f1d297619835d724db01904d",
        },
        type="Any2AnyRetrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{wu2021fashion,
  author = {Wu, Hui and Gao, Yupeng and Guo, Xiaoxiao and Al-Halah, Ziad and Rennie, Steven and Grauman, Kristen and Feris, Rogerio},
  booktitle = {Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
  pages = {11307--11317},
  title = {Fashion iq: A new dataset towards retrieving images by natural language feedback},
  year = {2021},
}
""",
        prompt={
            "query": "Find a fashion image that aligns with the reference image and style note."
        },
        superseded_by="FashionIQIT2IRetrieval.v2",
    )


class FashionIQIT2IRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FashionIQIT2IRetrieval.v2",
        description=(
            "Retrieve clothes based on descriptions. "
            "Version 2 sets the canonical metric to hit_rate_at_10, matching the "
            "M-BEIR/UniIR source metric (hit-style Recall@10) instead of "
            "ndcg_at_10. Dataset, corpus, and qrels are identical to FashionIQIT2IRetrieval. See "
            "[Issue #5214](https://github.com/embeddings-benchmark/mteb/issues/5214)."
        ),
        reference="https://openaccess.thecvf.com/content/CVPR2021/html/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.html",
        dataset={
            "path": "mteb/mbeir_fashioniq_task7",
            "revision": "469fba95d895f129f1d297619835d724db01904d",
        },
        type="Any2AnyRetrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{wu2021fashion,
  author = {Wu, Hui and Gao, Yupeng and Guo, Xiaoxiao and Al-Halah, Ziad and Rennie, Steven and Grauman, Kristen and Feris, Rogerio},
  booktitle = {Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
  pages = {11307--11317},
  title = {Fashion iq: A new dataset towards retrieving images by natural language feedback},
  year = {2021},
}
""",
        prompt={
            "query": "Find a fashion image that aligns with the reference image and style note."
        },
        adapted_from=["FashionIQIT2IRetrieval"],
    )
