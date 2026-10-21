from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetChartQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for ChartQA charts.

    The corpus contains chart images and the queries are multiple‑choice questions
    about those charts.  Each query is linked to one relevant chart image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetChartQA",
        description="Assess end-to-end vision-based RAG performance on real-world charts requiring complex logical and visual reasoning from retrieved images.",
        reference="https://aclanthology.org/2022.findings-acl.177/",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetChartQA",
            "revision": "4665d4a3ebc9ea5a4e635985a4fd2a595b78b3c0",
        },
        date=("2010-01-01", "2021-12-31"),
        domains=["Web", "Non-fiction"],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{masry2022chartqabenchmarkquestionanswering,
  address = {Dublin, Ireland},
  author = {Masry, Ahmed and Long, Do Xuan and Tan, Jia Qing and Joty, Shafiq and Hoque, Enamul},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2022},
  doi = {10.18653/v1/2022.findings-acl.177},
  editor = {Muresan, Smaranda and Nakov, Preslav and Villavicencio, Aline},
  month = {may},
  pages = {2263--2279},
  publisher = {Association for Computational Linguistics},
  title = {{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning},
  url = {https://aclanthology.org/2022.findings-acl.177/},
  year = {2022},
}""",
    )
