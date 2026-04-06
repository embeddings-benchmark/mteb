from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SDSKoPubVDRT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SDSKoPubVDRT2IRetrieval",
        description="SDS KoPub-VDR is a benchmark dataset for Visual Document Retrieval (VDR) in the context of Korean public documents. It contains real-world government document images paired with natural-language queries, corresponding answer pages, and ground-truth answers. The dataset is designed to evaluate AI models that go beyond simple text matching, requiring comprehensive understanding of visual layouts, tables, graphs, and images to accurately locate relevant information.",
        reference="https://arxiv.org/abs/2511.04910",
        dataset={
            "path": "whybe-choi/SDSKoPubVDRT2IRetrieval",
            "revision": "144fb16928ad9c552a3f556fa818130b2587c0bc",
        },
        type="DocumentUnderstanding",
        category="t2i",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=("2025-11-07", "2025-11-11"),
        domains=["Government", "Legal", "Non-fiction"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{lee2025sdskopubvdrbenchmark,
  archiveprefix = {arXiv},
  author = {Jaehoon Lee and Sohyun Kim and Wanggeun Park and Geon Lee and Seungkyung Kim and Minyoung Lee},
  eprint = {2511.04910},
  primaryclass = {cs.CL},
  title = {SDS KoPub VDR: A Benchmark Dataset for Visual Document Retrieval in Korean Public Documents},
  url = {https://arxiv.org/abs/2511.04910},
  year = {2025},
}
""",
        prompt={"query": "Find a screenshot that is relevant to the user's question."},
    )
