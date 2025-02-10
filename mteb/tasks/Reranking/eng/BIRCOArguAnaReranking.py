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


class BIRCOArguAnaReranking(AbsTaskReranking):
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

    def dataset_transform(self, hf_dataset):
        return birco_transform(hf_dataset, self.instruction)
