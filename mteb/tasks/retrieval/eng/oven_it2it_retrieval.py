from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class OVENIT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OVENIT2ITRetrieval",
        description="Retrieval a Wiki image and passage to answer query about an image.",
        reference="https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html",
        dataset={
            "path": "mteb/mbeir_oven_task8",
            "revision": "619847759581b2e57dfc1d68e2dadedc9599b283",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{hu2023open,
  author = {Hu, Hexiang and Luan, Yi and Chen, Yang and Khandelwal, Urvashi and Joshi, Mandar and Lee, Kenton and Toutanova, Kristina and Chang, Ming-Wei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages = {12065--12075},
  title = {Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
  year = {2023},
}
""",
        prompt={
            "query": "Retrieve a Wikipedia image-description pair that provides evidence for the question of this image."
        },
        superseded_by="OVENIT2ITRetrieval.v2",
    )


class OVENIT2ITRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="OVENIT2ITRetrieval.v2",
        description=(
            "Retrieval a Wiki image and passage to answer query about an image. "
            "Version 2 sets the canonical metric to hit_rate_at_5, matching the "
            "M-BEIR/UniIR source metric (hit-style Recall@5) instead of "
            "ndcg_at_10. Dataset, corpus, and qrels are identical to OVENIT2ITRetrieval. See "
            "[Issue #5214](https://github.com/embeddings-benchmark/mteb/issues/5214)."
        ),
        reference="https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Open-domain_Visual_Entity_Recognition_Towards_Recognizing_Millions_of_Wikipedia_Entities_ICCV_2023_paper.html",
        dataset={
            "path": "mteb/mbeir_oven_task8",
            "revision": "619847759581b2e57dfc1d68e2dadedc9599b283",
        },
        type="Any2AnyRetrieval",
        category="it2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_5",
        date=("2023-01-01", "2023-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{hu2023open,
  author = {Hu, Hexiang and Luan, Yi and Chen, Yang and Khandelwal, Urvashi and Joshi, Mandar and Lee, Kenton and Toutanova, Kristina and Chang, Ming-Wei},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages = {12065--12075},
  title = {Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities},
  year = {2023},
}
""",
        prompt={
            "query": "Retrieve a Wikipedia image-description pair that provides evidence for the question of this image."
        },
        adapted_from=["OVENIT2ITRetrieval"],
    )
