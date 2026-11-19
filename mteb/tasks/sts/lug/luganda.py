from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LugandaSTS(AbsTaskSTS):
    """Semantic textual similarity evaluation for Luganda sentence pairs."""

    metadata = TaskMetadata(
        name="LugandaSTS",
        description=(
            "Luganda Semantic Textual Similarity benchmark adapted from the "
            "STS Benchmark. It evaluates sentence embedding models by measuring "
            "the correlation between cosine similarity scores and the adapted "
            "semantic similarity scores for pairs of Luganda sentences. The "
            "current evaluation split contains 300 sentence pairs."
        ),
        dataset={
            "path": "sagepond/sts",
            "revision": "0d52f8af50b78a35d7134e6da422d3d04ca70606",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        reference="https://aclanthology.org/S17-2001/",
        eval_splits=["test"],
        eval_langs=["lug-Latn"],
        main_score="spearman",
        domains=["Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        date=("22-08-2026", "14-11-2026"),
        sample_creation="human-translated and localized",
        adapted_from=["STSBenchmark"],
        contributed_by="Caleb Matovu",
        is_public=True,
        bibtex_citation=r"""
        @inproceedings{cer-etal-2017-semeval,
    title = "{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation",
    author = "Cer, Daniel  and
      Diab, Mona  and
      Agirre, Eneko  and
      Lopez-Gazpio, I{\~n}igo  and
      Specia, Lucia",
    editor = "Bethard, Steven  and
      Carpuat, Marine  and
      Apidianaki, Marianna  and
      Mohammad, Saif M.  and
      Cer, Daniel  and
      Jurgens, David",
    booktitle = "Proceedings of the 11th {I}nternational {W}orkshop on {S}emantic {E}valuation ({S}em{E}val-2017)",
    month = aug,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S17-2001/",
    doi = "10.18653/v1/S17-2001",
    pages = "1--14",
    abstract = "Semantic Textual Similarity (STS) measures the meaning similarity of sentences. Applications include machine translation (MT), summarization, generation, question answering (QA), short answer grading, semantic search, dialog and conversational systems. The STS shared task is a venue for assessing the current state-of-the-art. The 2017 task focuses on multilingual and cross-lingual pairs with one sub-track exploring MT quality estimation (MTQE) data. The task obtained strong participation from 31 teams, with 17 participating in \textit{all language tracks}. We summarize performance and review a selection of well performing methods. Analysis highlights common errors, providing insight into the limitations of existing models. To support ongoing work on semantic representations, the \textit{STS Benchmark} is introduced as a new shared training and evaluation set carefully selected from the corpus of English STS shared task data (2012-2017)."
}
        """,
    )

    column_names = ("sentence1", "sentence2")
    min_score = 0
    max_score = 5
