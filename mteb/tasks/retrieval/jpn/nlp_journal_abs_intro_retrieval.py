from mteb.abstasks.retrieval import AbsTaskRetrieval
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
            "revision": "8a7ae2cbcd62a3df5a880f4e4afff30a977afe0d",
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
@inproceedings{li-etal-2026-jmteb,
  address = {Palma de Mallorca, Spain},
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  doi = {10.63317/5ouzpv2f2f6k},
  month = may,
  pages = {7423--7434},
  publisher = {ELRA Language Resource Association},
  title = {{JMTEB} and {JMTEB}-lite: {J}apanese Massive Text Embedding Benchmark and Its Lightweight Version},
  url = {https://aclanthology.org/2026.lrec-1.588/},
  year = {2026},
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
            "revision": "e1b0d0b036b080a1b4642f5dd2e5f4e983e73d54",
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
@inproceedings{li-etal-2026-jmteb,
  address = {Palma de Mallorca, Spain},
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide and Kawahara, Daisuke},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference},
  doi = {10.63317/5ouzpv2f2f6k},
  month = may,
  pages = {7423--7434},
  publisher = {ELRA Language Resource Association},
  title = {{JMTEB} and {JMTEB}-lite: {J}apanese Massive Text Embedding Benchmark and Its Lightweight Version},
  url = {https://aclanthology.org/2026.lrec-1.588/},
  year = {2026},
}
""",
    )
