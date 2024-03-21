from __future__ import annotations

from collections import defaultdict

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    corpus_dataset = load_dataset(hf_hub_name, "corpus")
    queries_dataset = load_dataset(hf_hub_name, "queries")
    qrels = load_dataset(hf_hub_name + "-qrels")[eval_split]

    corpus = {e["_id"]: {"text": e["text"]} for e in corpus_dataset["corpus"]}
    queries = {e["_id"]: e["text"] for e in queries_dataset["queries"]}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["query-id"]][e["corpus-id"]] = e["score"]

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs


class GermanQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GermanQuAD-Retrieval",
        description="Context Retrieval for German Question Answering",
        reference="https://www.kaggle.com/datasets/GermanQuAD",
        hf_hub_name="mteb/germanquad",
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["de"],
        main_score="mrr_at_5",
        revision="f5c87ae5a2e7a5106606314eef45255f03151bb3",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )


    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["hf_hub_name"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True
