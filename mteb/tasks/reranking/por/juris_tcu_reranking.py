from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JurisTCUReranking(AbsTaskRetrieval):
    """Genuine JurisTCU reranking — rerank BM25 top-100 candidates per query."""

    metadata = TaskMetadata(
        name="JurisTCUReranking",
        description=(
            "Genuine legal-domain reranking: for each of 150 queries, rerank the "
            "~100 first-stage candidates (BM25 top-100 over the full 16k-document "
            "TCU jurisprudence corpus, unioned with human-judged docs) under "
            "graded relevance 0-3. Candidates include lexical hard negatives, so "
            "the task is distinct from first-stage retrieval. Complements "
            "QuatiReranking (web) with a legal-domain reranking probe."
        ),
        reference="https://huggingface.co/datasets/LeandroRibeiro/JurisTCU",
        dataset={
            "path": "MTEB-BR/JurisTCUReranking",
            "revision": "9f9bcc649c05819d3c8055d7a442568bed68a83e",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="map_at_1000",
        date=("2020-01-01", "2025-12-31"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        sample_creation="created",
        adapted_from=["JurisTCU"],
        bibtex_citation=r"""
@article{juristcu2026,
  author = {Fernandes, Leandro Car{\'i}sio and
                 Ribeiro, Leandro dos Santos and
                 de Castro, Marcos Vin{\'i}cius Borela and
                 da Silva Pacheco, Leonardo Augusto and
                 de Oliveira Sandes, Edans Fl{\'a}vius},
  title = {{JurisTCU: a Brazilian Portuguese information retrieval dataset with query relevance judgments}},
  journal = {Language Resources and Evaluation},
  year = {2026},
  volume = {60},
  number = {1},
  doi = {10.1007/s10579-025-09881-w},
  url = {https://doi.org/10.1007/s10579-025-09881-w},
  issn = {1574-0218},
}
""",
    )
