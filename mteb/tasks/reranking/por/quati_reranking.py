from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class QuatiReranking(AbsTaskRetrieval):
    """Genuine Quati reranking — rerank BM25 top-100 candidates per query."""

    metadata = TaskMetadata(
        name="QuatiReranking",
        description=(
            "Genuine web-domain reranking: for each of 50 PT-BR queries, rerank "
            "the ~100 first-stage candidates (BM25 top-100 over the full "
            "1M-passage Quati Brazilian web corpus, unioned with human-judged "
            "passages) under graded relevance 0-3. Candidates include lexical "
            "hard negatives, so the task is distinct from first-stage retrieval. "
            "Complements JurisTCUReranking (legal) with a web-domain probe."
        ),
        reference="https://aclanthology.org/2024.stil-1.19/",
        dataset={
            "path": "MTEB-BR/QuatiReranking",
            "revision": "c8a302b7248d659b14d83ce4e3862e4b7b9cf0a5",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="map_at_1000",
        date=("2022-01-01", "2024-06-30"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        adapted_from=["Quati"],
        bibtex_citation=r"""
@inproceedings{bueno-etal-2024-quati,
  title = {{Q}uati: A {B}razilian {P}ortuguese Information Retrieval Dataset from Native Speakers},
  author = {Bueno, Mirelle and Maia Sanches, Eduardo and Lotufo, Roberto and Nogueira, Rodrigo},
  booktitle = {Proceedings of the 16th Symposium in Information and Human Language Technology (STIL 2024)},
  year = {2024},
  url = {https://aclanthology.org/2024.stil-1.19/},
}
""",
    )
