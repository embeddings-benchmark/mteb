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


class BIRCORelicReranking(AbsTaskReranking):
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

    def dataset_transform(self, hf_dataset):
        return birco_transform(hf_dataset, self.instruction)
