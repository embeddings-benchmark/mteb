from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MMVUVideoCentricQA(AbsTaskRetrieval):
    k_values = (1, 3, 5)  # 5 possible answers only

    metadata = TaskMetadata(
        name="MMVUVideoCentricQA",
        description="MMVU is an expert-level, multi-discipline video understanding benchmark with questions spanning 27 subjects across Science, Healthcare, Humanities & Social Sciences, and Engineering. Each multiple-choice example pairs a specialized-domain video with a question and 5 candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate. Used the public validation multiple-choice subset (~625 examples).",
        reference="https://arxiv.org/abs/2501.12380",
        dataset={
            "path": "mteb/MMVU-VQA",
            "revision": "4c2b59cb04639eacbae2f6a2379e538d5696ab7a",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2025-01-21", "2025-01-21"),
        domains=["Academic", "Medical", "Engineering", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{zhao2025mmvu,
  author = {Zhao, Yilun and Zhang, Haowei and Xie, Lujing and Hu, Tongyan and Gan, Guo and Long, Yitao and Hu, Zhiyuan and Chen, Weiyuan and Li, Chuhan and Xu, Zhijian and others},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages = {8475--8489},
  title = {MMVU: Measuring expert-level multi-discipline video understanding},
  year = {2025},
}
""",
    )
