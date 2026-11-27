from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FaqBacenRetrieval(AbsTaskRetrieval):
    """Retrieve the correct BACEN regulatory answer for a citizen question."""

    metadata = TaskMetadata(
        name="FaqBacenRetrieval",
        description=(
            "Retrieve the correct answer to a citizen question about Brazilian "
            "financial/banking regulation, from the Banco Central do Brasil "
            "(BACEN) public FAQ. 373 test questions over a pool of 1673 unique "
            "regulatory answers. Native PT-BR; financial/government domain."
            " Retrieval formulation constructed for MTEB-BR (Stekel, 2026) from the Itau-Unibanco/FAQ_BACEN dataset."
        ),
        reference="https://huggingface.co/datasets/Itau-Unibanco/FAQ_BACEN",
        dataset={
            "path": "MTEB-BR/FaqBacenRetrieval",
            "revision": "179270c3ef79ccf1d2c03adc2a80830716c574ff",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),
        domains=["Financial", "Government", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=["pt-BR"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{stekel2026mtebbr,
  archiveprefix = {arXiv},
  author = {Stekel, Tardelli Ronan Coelho},
  eprint = {2607.04581},
  primaryclass = {cs.CL},
  title = {{MTEB-BR}: A Text Embedding Benchmark for {B}razilian {P}ortuguese},
  url = {https://arxiv.org/abs/2607.04581},
  year = {2026},
}

@misc{faq_bacen,
  author = {{Ita\'u Unibanco}},
  howpublished = {HuggingFace dataset \texttt{Itau-Unibanco/FAQ\_BACEN}, sourced from the Banco Central do Brasil FAQ},
  title = {{FAQ\_BACEN} (Hugging Face dataset)},
  url = {https://huggingface.co/datasets/Itau-Unibanco/FAQ_BACEN},
}
""",
    )
