# BIRCO_Reranking.py

from ....abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.TaskMetadata import TaskMetadata

class BIRCORerankingBase(AbsTaskReranking):
    """
    Base class for all BIRCO reranking tasks.
    
    Expects the Hugging Face dataset to have records with keys:
      - "query_id": str
      - "query_text": str
      - "corpus": list of dicts (each with "corpus_id" and "corpus_text")
      - "relevance": dict mapping corpus_id to an integer relevance score
    Also uses an instruction stored in metadata (metadata.instruction).
    """
    def dataset_transform(self, hf_dataset):
        queries = {}
        corpus = {}
        qrels = {}
        instructions = {}
        
        # For each record, build the standard dictionaries.
        for record in hf_dataset:
            qid = record["query_id"]
            queries[qid] = record["query_text"]
            qrels[qid] = record["relevance"]
            # Use the instruction defined in metadata (if provided); otherwise, use an empty string.
            instructions[qid] = getattr(self.metadata, "instruction", "")
            for doc in record["corpus"]:
                cid = doc["corpus_id"]
                if cid not in corpus:
                    corpus[cid] = doc["corpus_text"]
                    
        return {"queries": queries, "corpus": corpus, "qrels": qrels, "instructions": instructions}


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
        category="Academic",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg",
        dataset={"path": "mteb/BIRCO-DorisMae-Test", "revision": "latest"},
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=["Information Retrieval", "Complex Query"],
        license="cc-by-4.0",
        annotations_creators="BIRCO authors",
        dialect=[],
        sample_creation="extracted",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}},
}"""
    )
    # Add the task-specific instruction to metadata.
    metadata.instruction = (
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
        category="Debate",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg",
        dataset={"path": "mteb/BIRCO-ArguAna-Test", "revision": "latest"},
        date=("2024-01-01", "2024-12-31"),
        domains=["Politics", "Debate"],
        task_subtypes=["Information Retrieval", "Counterargument Retrieval"],
        license="cc-by-4.0",
        annotations_creators="BIRCO authors",
        dialect=[],
        sample_creation="extracted",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}},
}"""
    )
    metadata.instruction = (
        "Given a one-paragraph argument, retrieve the passage that contains the counter-argument which directly refutes the query's stance."
    )


class BIRCOClinicalTrialReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-ClinicalTrial",
        description=(
            "Retrieval task using the Clinical-Trial dataset from BIRCO. This dataset contains 50 queries that are patient case reports. "
            "Each query has a candidate pool comprising 30-110 clinical trial descriptions. Relevance is graded (0, 1, 2), where 1 and 2 are relevant."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="Biomedical",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg",
        dataset={"path": "mteb/BIRCO-ClinicalTrial-Test", "revision": "latest"},
        date=("2024-01-01", "2024-12-31"),
        domains=["Biomedical"],
        task_subtypes=["Information Retrieval", "Clinical Trial Matching"],
        license="cc-by-4.0",
        annotations_creators="BIRCO authors",
        dialect=[],
        sample_creation="extracted",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}},
}"""
    )
    metadata.instruction = (
        "Given a patient case report, retrieve the clinical trial description that best matches the patient's eligibility criteria."
    )


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
        category="Literature",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg",
        dataset={"path": "mteb/BIRCO-WTB-Test", "revision": "latest"},
        date=("2024-01-01", "2024-12-31"),
        domains=["Literature"],
        task_subtypes=["Information Retrieval", "Ambiguous Query Resolution"],
        license="cc-by-4.0",
        annotations_creators="BIRCO authors",
        dialect=[],
        sample_creation="extracted",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}},
}"""
    )
    metadata.instruction = (
        "Given an ambiguous description of a book, retrieve the book description that best matches the query."
    )


class BIRCORelicReranking(BIRCORerankingBase):
    metadata = TaskMetadata(
        name="BIRCO-Relic",
        description=(
            "Retrieval task using the RELIC dataset from BIRCO. This dataset contains 100 queries that are excerpts from literary analyses "
            "with a missing quotation (indicated by [masked sentence(s)]). Each query has a candidate pool of 50 passages. "
            "The objective is to retrieve the passage that best completes the literary analysis."
        ),
        reference="https://github.com/BIRCO-benchmark/BIRCO",
        type="Reranking",
        category="Literature",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg",
        dataset={"path": "mteb/BIRCO-Relic-Test", "revision": "latest"},
        date=("2024-01-01", "2024-12-31"),
        domains=["Literature"],
        task_subtypes=["Information Retrieval", "Quotation Retrieval"],
        license="cc-by-4.0",
        annotations_creators="BIRCO authors",
        dialect=[],
        sample_creation="extracted",
        bibtex_citation="""@misc{BIRCO,
  title={BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives},
  author={Wang et al.},
  year={2024},
  howpublished={\\url{https://github.com/BIRCO-benchmark/BIRCO}},
}"""
    )
    metadata.instruction = (
        "Given a literary analysis with a missing quotation (marked as [masked sentence(s)]), retrieve the passage that best completes the analysis."
    )
