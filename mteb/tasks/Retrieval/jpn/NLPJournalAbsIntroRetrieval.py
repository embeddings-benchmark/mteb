from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLIT = "test"


class NLPJournalAbsIntroRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NLPJournalAbsIntroRetrieval.V2",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding introduction with the given abstract. "
            "This is the V2 dataset (last update 2025-06-15)."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/NLPJournalAbsIntroRetrieval.V2",
            "revision": "441779db89bd963e76abecd2d8b2165fcb1a9134",
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
        adapted_from=["NLPJournalAbsIntroRetrieval"],
        bibtex_citation=r"""
@misc{jmteb,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB}},
  title = {{J}{M}{T}{E}{B}: {J}apanese {M}assive {T}ext {E}mbedding {B}enchmark},
  year = {2024},
}
""",
    )


class NLPJournalAbsIntroRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NLPJournalAbsIntroRetrieval",
        description=(
            "This dataset was created from the Japanese NLP Journal LaTeX Corpus. "
            "The titles, abstracts and introductions of the academic papers were shuffled. "
            "The goal is to find the corresponding introduction with the given abstract. "
            "This is the V1 dataset (last update 2020-06-15)."
        ),
        reference="https://huggingface.co/datasets/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/NLPJournalAbsIntroRetrieval",
            "revision": "c3cc9f3ae48454195ca5663ef463b2bcc39b2dff",
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
