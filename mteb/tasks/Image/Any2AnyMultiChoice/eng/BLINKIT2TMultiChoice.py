from collections import defaultdict

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BLINKIT2TMultiChoice(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BLINKIT2TMultiChoice",
        description="Retrieve the correct text answer based on images and specific retrieval instructions.",
        reference="https://arxiv.org/abs/2404.12390",
        dataset={
            "path": "JamieSJS/blink-it2t-multi",
            "revision": "bc8f4c7f62450a4ceb737c8339061cf87aea42d5",
        },
        type="VisionCentricQA",
        category="it2t",
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
@article{fu2024blink,
  author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
  journal = {arXiv preprint arXiv:2404.12390},
  title = {Blink: Multimodal large language models can see but not perceive},
  year = {2024},
}
""",
    )

    def dataset_transform(self, **kwargs):
        for subset, split_data in self.dataset.items():
            for split, dataset in split_data.items():
                top_ranked = defaultdict(list)
                for query_id, relevant in dataset["relevant_docs"].items():
                    for corpus_id, score in relevant.items():
                        top_ranked[query_id].append(corpus_id)
                dataset["top_ranked"] = top_ranked
