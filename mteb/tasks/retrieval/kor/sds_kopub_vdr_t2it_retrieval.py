from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SDSKoPubVDRT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SDSKoPubVDRT2ITRetrieval",
        description="SDS KoPub-VDR is a benchmark dataset for Visual Document Retrieval (VDR) in the context of Korean public documents. It contains real-world government document images paired with natural-language queries, corresponding answer pages, and ground-truth answers. This multimodal retrieval task provides both PyPDF-extracted text and page images as the corpus, enabling evaluation of text, image, and multimodal retrieval models.",
        reference="https://arxiv.org/abs/2511.04910",
        dataset={
            "path": "whybe-choi/SDSKoPubVDRT2ITRetrieval",
            "revision": "035bd023faba4af3be283b97413bc92cf791ac4f",
        },
        type="DocumentUnderstanding",
        category="t2it",
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
        prompt={
            "query": "Find a document page that is relevant to the user's question."
        },
    )
