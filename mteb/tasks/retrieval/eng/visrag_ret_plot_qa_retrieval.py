from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VisRAGRetPlotQA(AbsTaskRetrieval):
    """VisRAG Retrieval task for PlotQA scientific plots.

    The corpus contains scientific plot images and the queries are questions
    requiring reasoning over the plots.  Each query has exactly one relevant
    plot image.
    """

    metadata = TaskMetadata(
        name="VisRAGRetPlotQA",
        description="Execute vision-based retrieval and numerical reasoning over scientific plots to answer questions without relying on structured data parsing.",
        reference="https://arxiv.org/abs/1909.00997",
        type="Retrieval",
        task_subtypes=["Image Text Retrieval"],
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        dataset={
            "path": "mteb/VisRAGRetPlotQA",
            "revision": "0c2dad65a167e80a131c4839211ca8ec5b7f870e",
        },
        date=("2000-01-01", "2019-12-31"),
        domains=["Web", "Non-fiction"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{methani2020plotqareasoningscientificplots,
  archiveprefix = {arXiv},
  author = {Nitesh Methani and Pritha Ganguly and Mitesh M. Khapra and Pratyush Kumar},
  eprint = {1909.00997},
  primaryclass = {cs.CV},
  title = {PlotQA: Reasoning over Scientific Plots},
  url = {https://arxiv.org/abs/1909.00997},
  year = {2020},
}""",
    )
