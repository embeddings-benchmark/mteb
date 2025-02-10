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


class BIRCODorisMaeReranking(AbsTaskReranking):
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

    def dataset_transform(self, hf_dataset):
        return birco_transform(hf_dataset, self.instruction)
