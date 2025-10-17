from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class NLPJournalTitleIntroRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NLPJournalTitleIntroRetrieval.V2",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding introduction with the given title. "
            "This is the V2 dataset (last updated 2025-06-15)."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/NLPJournalTitleIntroRetrieval.V2",
            "revision": "39b53c8298ce1948d2a4caaa55fee5b34eeb647e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("1994-10-10", "2025-06-15"),
        domains=["Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        adapted_from=["NLPJournalTitleIntroRetrieval"],
        bibtex_citation=r"""
@misc{jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
  title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
  year = {2024},
}
""",
    )


class NLPJournalTitleIntroRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NLPJournalTitleIntroRetrieval",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding introduction with the given title. "
            "This is the V1 dataset (last updated 2020-06-15)."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/NLPJournalTitleIntroRetrieval",
            "revision": "4c58ad0940d449fa5d71ce4ec285502402671a5b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("1994-10-10", "2020-06-15"),
        domains=["Academic", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
  title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
  year = {2024},
}
""",
    )
