from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BLINKIT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BLINKIT2IRetrieval",
        description="Retrieve images based on images and specific retrieval instructions.",
        reference="https://arxiv.org/abs/2404.12390",
        dataset={
            "path": "JamieSJS/blink-it2i",
            "revision": "7a1a1330565faca9c1aeec6f5acfc64f21296753",
        },
        type="Any2AnyRetrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_1",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{fu2024blink,
  author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
  journal = {arXiv preprint arXiv:2404.12390},
  title = {Blink: Multimodal large language models can see but not perceive},
  year = {2024},
}
""",
    )
