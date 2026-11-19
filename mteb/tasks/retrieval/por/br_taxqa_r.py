from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BRTaxQAR(AbsTaskRetrieval):
    """BR-TaxQA-R — Brazilian Portuguese tax-law retrieval, 715 queries / 478 docs."""

    metadata = TaskMetadata(
        name="BRTaxQAR",
        description=(
            "BR-TaxQA-R: Brazilian Portuguese tax-law information retrieval. 715 "
            "questions about Brazilian tax law paired with 478 legal documents "
            "(Leis, Decretos, Instruções Normativas, etc.). Relevance judgments "
            "are derived from the explicit legal citations in each answer: "
            "primary references (score 2) and embedded references mentioned "
            "within the answer text (score 1). Complements JurisTCU's "
            "jurisprudence retrieval with statutory tax-code retrieval."
        ),
        reference="https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R",
        dataset={
            "path": "MTEB-BR/BRTaxQAR",
            "revision": "83694d760fe63741e348ef80b37f420313f9096d",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Legal", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{junior-etal-2025-brtaxqar,
  author = {J{\'u}nior, Juvenal Domingos and Faria, Augusto and de Oliveira, E. Seiti and de Brito, Erick and Teotonio, Matheus and Assump{\c{c}}{\~a}o, Andre and Carmo, Diedre and Lotufo, Roberto and Pereira, Jayr},
  booktitle = {Intelligent Systems, {BRACIS} 2025},
  doi = {10.1007/978-3-032-15987-8_14},
  pages = {208--222},
  publisher = {Springer},
  title = {{BR-TaxQA-R}: A Dataset for Question Answering with References for {B}razilian Personal Income Tax Law, Including Case Law},
  url = {https://sol.sbc.org.br/index.php/bracis/article/view/40777},
  year = {2025},
}
""",
    )
