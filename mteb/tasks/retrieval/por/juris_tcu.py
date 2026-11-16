from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JurisTCU(AbsTaskRetrieval):
    """JurisTCU — Brazilian Portuguese legal retrieval, 150 queries / 16k docs."""

    metadata = TaskMetadata(
        name="JurisTCU",
        description=(
            "JurisTCU: Brazilian Portuguese legal IR over Tribunal de Contas da União "
            "jurisprudence. 150 queries (mix of real search log + LLM-derived) with "
            "2,250 graded relevance judgments (0-3 scale) over 16,045 curated "
            "jurisprudence documents from the TCU 'Selected Jurisprudence' collection. "
            "Tests legal-domain retrieval in PT-BR, complementing Quati's general-web "
            "domain."
        ),
        reference="https://huggingface.co/datasets/LeandroRibeiro/JurisTCU",
        dataset={
            "path": "MTEB-BR/JurisTCU",
            "revision": "808f8e1041a63ba0d6d98521190688f3bede2441",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-12-31"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
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
