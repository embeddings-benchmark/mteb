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


class BIRCOClinicalTrialReranking(AbsTaskReranking):
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

    def dataset_transform(self, hf_dataset):
        return birco_transform(hf_dataset, self.instruction)
