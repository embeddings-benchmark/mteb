from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)

_SUBSETS = [
    "foil",
    "answer_importance",
    "brevity_bias",
    "literal_bias",
    "poison",
    "position_bias",
    "repetition_bias",
]

# For each subset, select the candidate that should rank first.
# foil: doc2 has evidence, doc1 is foil
# poison: doc2 has true evidence, doc1 is poisoned
# answer_importance: doc1 has evidence, doc2 has no evidence
# In the four individual-bias subsets, both documents contain the answer. MTEB
# needs a single target to score the pair, so doc2 is the less-biased control;
# accuracy for those subsets measures resistance to the named bias rather than
# ordinary relevance.
_PREFERRED_DOC = {
    "foil": "doc2",
    "poison": "doc2",
    "answer_importance": "doc1",
    "brevity_bias": "doc2",
    "literal_bias": "doc2",
    "position_bias": "doc2",
    "repetition_bias": "doc2",
}


class ColDeRReranking(AbsTaskRetrieval):
    """ColDeR: Collapse of Dense Retrievers - Short, Early, and Literal Biases Outranking Factual Evidence."""

    metadata = TaskMetadata(
        name="ColDeRReranking",
        description=(
            "ColDeR (Collapse of Dense Retrievers) evaluates retriever vulnerability to "
            "heuristic biases (brevity, position, repetition, literal matching) and failure "
            "modes where biased non-evidence documents outrank factual evidence. For "
            "the individual-bias subsets where both candidates contain evidence, accuracy "
            "measures preference for the less-biased control document."
        ),
        reference="https://arxiv.org/abs/2503.05037",
        dataset={
            "path": "mohsenfayyaz/ColDeR",
            "revision": "368a695c5593f5ee7fdd741706caad8dcf4696ac",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={s: ["eng-Latn"] for s in _SUBSETS},
        main_score="accuracy",
        date=("2025-01-01", "2025-05-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Reasoning as Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{fayyaz-etal-2025-collapse,
  address = {Vienna, Austria},
  author = {Fayyaz, Mohsen  and
Modarressi, Ali  and
Schuetze, Hinrich  and
Peng, Nanyun},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month = jul,
  pages = {9136--9152},
  publisher = {Association for Computational Linguistics},
  title = {Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence},
  url = {https://aclanthology.org/2025.acl-long.447/},
  year = {2025},
}
""",
        prompt={
            "query": "Given a question, retrieve the document that provides the factual answer."
        },
    )

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        if self.data_loaded:
            return

        self.dataset = {}
        for subset in self.hf_subsets:
            ds = load_dataset(
                self.metadata.dataset["path"],
                data_files={"test": f"test/{subset}.jsonl"},
                split="test",
                revision=self.metadata.dataset.get("revision"),
            )
            queries = []
            corpus = []
            relevant_docs = {}
            top_ranked = {}

            preferred_doc = _PREFERRED_DOC[subset]

            for i, row in enumerate(ds):
                qid = f"{subset}_{i}"
                doc1_id = f"{qid}_doc1"
                doc2_id = f"{qid}_doc2"

                queries.append({"id": qid, "text": row["query"]})
                corpus.append({"id": doc1_id, "text": row["document_1"], "title": ""})
                corpus.append({"id": doc2_id, "text": row["document_2"], "title": ""})

                preferred_id = doc1_id if preferred_doc == "doc1" else doc2_id
                relevant_docs[qid] = {preferred_id: 1}
                top_ranked[qid] = [doc1_id, doc2_id]

            self.dataset[subset] = {
                "test": {
                    "queries": Dataset.from_list(queries),
                    "corpus": Dataset.from_list(corpus),
                    "relevant_docs": relevant_docs,
                    "top_ranked": top_ranked,
                }
            }

        self.data_loaded = True
