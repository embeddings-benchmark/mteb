from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NQVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ-VN",
        dataset={
            "path": "GreenNode/nq-vn",
            "revision": "40a6d7f343b9c9f4855a426d8c431ad5f8aaf56b",
        },
        description="A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        bibtex_citation=r"""
@inproceedings{pham2025vnmtebvietnamesemassivetext,
  address = {Rabat, Morocco},
  author = {Pham, Loc and Luu, Tung and Vo, Thu and Nguyen, Minh and Hoang, Viet},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026},
  doi = {10.18653/v1/2026.findings-eacl.86},
  editor = {Demberg, Vera and Inui, Kentaro and Marquez, Llu{\'i}s},
  isbn = {979-8-89176-386-9},
  month = {mar},
  pages = {1705--1725},
  publisher = {Association for Computational Linguistics},
  title = {{VN}-{MTEB}: {V}ietnamese Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2026.findings-eacl.86/},
  year = {2026},
}
""",
        adapted_from=["NQ"],
    )


class NanoNQVN(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NanoNQ-VN",
        dataset={
            "path": "GreenNode/nano-nq-vn",
            "revision": "1ad4d6556fe0e5314994839089ce070fb0db8b19",
        },
        description="NanoNQVN is a small version of A translated dataset from NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval The process of creating the VN-MTEB (Vietnamese Massive Text Embedding Benchmark) from English samples involves a new automated system: - The system uses large language models (LLMs), specifically Coherence's Aya model, for translation. - Applies advanced embedding models to filter the translations. - Use LLM-as-a-judge to scoring the quality of the samples base on multiple criteria.",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="t2t",
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and LM verified",
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        bibtex_citation=r"""
@inproceedings{pham2025vnmtebvietnamesemassivetext,
  address = {Rabat, Morocco},
  author = {Pham, Loc and Luu, Tung and Vo, Thu and Nguyen, Minh and Hoang, Viet},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026},
  doi = {10.18653/v1/2026.findings-eacl.86},
  editor = {Demberg, Vera and Inui, Kentaro and Marquez, Llu{\'i}s},
  isbn = {979-8-89176-386-9},
  month = {mar},
  pages = {1705--1725},
  publisher = {Association for Computational Linguistics},
  title = {{VN}-{MTEB}: {V}ietnamese Massive Text Embedding Benchmark},
  url = {https://aclanthology.org/2026.findings-eacl.86/},
  year = {2026},
}
""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NQ-VN"],
    )
