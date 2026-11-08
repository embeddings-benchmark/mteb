from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BLINKIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BLINKIT2TRetrieval",
        description="Retrieve images based on images and specific retrieval instructions.",
        reference="https://doi.org/10.1007/978-3-031-73337-6_9",
        dataset={
            "path": "mteb/blink-it2t",
            "revision": "1ce9401bf888d851ee1b221dbb6bec718128ce00",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_1",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inbook{fu2024blink,
  author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A. and Ma, Wei-Chiu and Krishna, Ranjay},
  booktitle = {Computer Vision – ECCV 2024},
  doi = {10.1007/978-3-031-73337-6_9},
  isbn = {9783031733376},
  issn = {1611-3349},
  month = {Oct},
  pages = {148–166},
  publisher = {Springer Nature Switzerland},
  title = {BLINK: Multimodal Large Language Models Can See but Not Perceive},
  url = {https://doi.org/10.1007/978-3-031-73337-6_9},
  year = {2024},
}
""",
    )
