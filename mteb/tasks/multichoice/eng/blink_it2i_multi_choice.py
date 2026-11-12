from collections import defaultdict
from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BLINKIT2IMultiChoice(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BLINKIT2IMultiChoice",
        description="Retrieve images based on images and specific retrieval instructions.",
        reference="https://doi.org/10.1007/978-3-031-73337-6_9",
        dataset={
            "path": "mteb/blink-it2i-multi",
            "revision": "0418ecbea812ab97bda7eabe2c34065fdb81810f",
        },
        type="VisionCentricQA",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="not specified",
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

    def dataset_transform(self, **kwargs: Any):
        for split_data in self.dataset.values():
            for dataset in split_data.values():
                top_ranked = defaultdict(list)
                for query_id, relevant in dataset["relevant_docs"].items():
                    for corpus_id in relevant:
                        top_ranked[query_id].append(corpus_id)
                dataset["top_ranked"] = top_ranked
