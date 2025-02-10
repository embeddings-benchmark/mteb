from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


# Optional: Extract the transformation logic as a function.
def birco_transform(hf_dataset, default_instruction=""):
    queries = {}
    corpus = {}
    qrels = {}
    instructions = {}
    for record in hf_dataset:
        qid = record["query_id"]
        queries[qid] = record["query_text"]
        qrels[qid] = record["relevance"]
        instructions[qid] = default_instruction
        for doc in record["corpus"]:
            cid = doc["corpus_id"]
            if cid not in corpus:
                corpus[cid] = doc["corpus_text"]
    return {
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels,
        "instructions": instructions,
    }


class BIRCORerankingBase(AbsTaskReranking):
    """Base class for BIRCO reranking tasks.
    Uses a standard dataset transformation for BIRCO tasks.
    """

    def dataset_transform(self, hf_dataset):
        default_inst = getattr(self, "instruction", "")
        return birco_transform(hf_dataset, default_inst)


class BIRCODorisMaeReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-DorisMae",
        description=(
            "Retrieval task using the DORIS-MAE dataset from BIRCO. This dataset contains 60 queries "
            "that are complex research questions from computer scientists. Each query has a candidate pool of "
            "approximately 110 abstracts. Relevance is graded from 0 to 2 (scores of 1 and 2 are considered relevant)."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",  # Valid category (sentence-to-paragraph)
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-DorisMae-Test",
            "revision": "27d9d0022ce22cc770ad0c6cefaf26674d5eb399",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic"],  # TASK_DOMAIN accepts "Academic"
        task_subtypes=["Scientific Reranking"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",  # Valid annotator type
        dialect=[],
        sample_creation="found",  # Valid sample creation method
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}}
}""",
    )
    instruction = (
        "The query communicates specific research requirements. "
        "Identify the abstract that best fulfills the research requirements described in the query."
    )


class BIRCOArguAnaReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-ArguAna",
        description=(
            "Retrieval task using the ArguAna dataset from BIRCO. This dataset contains 100 queries where both queries "
            "and passages are complex one-paragraph arguments about current affairs. The objective is to retrieve the counter-argument "
            "that directly refutes the query’s stance."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-ArguAna-Test",
            "revision": "76f66dcb0253bcacbbfeddce2a53041a765e048c",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Written"],  # there is no 'Debate' domain
        task_subtypes=["Reasoning as Retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}}
}""",
    )
    instruction = "Given a one-paragraph argument, retrieve the passage that contains the counter-argument which directly refutes the query's stance."


class BIRCOClinicalTrialReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-ClinicalTrial",
        description=(
            "Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. "
            "Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are considered relevant."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-ClinicalTrial-Test",
            "revision": "4f616dc0f2349ba3be31f3202ee4f3baef6438b6",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Medical"],  # Valid domain (Medical)
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}}
}""",
    )
    instruction = "Given a patient case report, retrieve the clinical trial description that best matches the patient's eligibility criteria."


class BIRCOWhatsThatBookReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-WTB",
        description=(
            "Retrieval task using the WhatsThatBook dataset from BIRCO. This dataset contains 100 queries where each query "
            "is an ambiguous description of a book. Each query has a candidate pool of 50 book descriptions. "
            "The objective is to retrieve the correct book description."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-WTB-Test",
            "revision": "acf9fc30a976378e7cd17a9c3f6c065c2b76e4b5",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Fiction"],  # Valid domain (Fiction)
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}}
}""",
    )
    instruction = "Given an ambiguous description of a book, retrieve the book description that best matches the query."


class BIRCORelicReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-Relic",
        description=(
            "Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries which are excerpts from literary analyses "
            "with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. "
            "The objective is to retrieve the passage that best completes the literary analysis."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "mteb/BIRCO-Relic-Test",
            "revision": "f1f127af9f445ec706769f8491ea663525bb5c93",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Fiction"],  # Valid domain
        task_subtypes=["Article retrieval"],  # Valid subtype
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}}
}""",
    )
    instruction = "Given a literary analysis with a missing quotation (marked as [masked sentence(s)]), retrieve the passage that best completes the analysis."
