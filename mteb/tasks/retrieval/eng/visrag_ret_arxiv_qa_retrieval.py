from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetArxivQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VisRAGRetArxivQA",
        description="evaluate vision-based retrieval and generation on scientific figures and their surrounding context to preserve complex layouts and mathematical notations.",
        reference="https://arxiv.org/abs/2403.00231",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetArxivQA",
            "revision": "b0da0f3f9677461eb78e34a7164596ff4f86bd52",
        },
        date=("2000-01-01", "2024-12-31"),
        domains=["Academic", "Non-fiction"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{li2024multimodalarxivdatasetimproving,
  archiveprefix = {arXiv},
  author = {Lei Li and Yuqi Wang and Runxin Xu and Peiyi Wang and Xiachong Feng and Lingpeng Kong and Qi Liu},
  eprint = {2403.00231},
  primaryclass = {cs.CV},
  title = {Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models},
  url = {https://arxiv.org/abs/2403.00231},
  year = {2024},
}""",
    )
