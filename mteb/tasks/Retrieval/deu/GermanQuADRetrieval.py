from __future__ import annotations

from collections import defaultdict

from datasets import DatasetDict, load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


def load_retrieval_data(dataset_path, eval_splits):
    eval_split = eval_splits[0]
    corpus_dataset = load_dataset(dataset_path, "corpus")
    queries_dataset = load_dataset(dataset_path, "queries")
    qrels = load_dataset(dataset_path + "-qrels")[eval_split]

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
        reference="https://huggingface.co/datasets/deepset/germanquad",
        dataset={
            "path": "mteb/germanquad-retrieval",
            "revision": "f5c87ae5a2e7a5106606314eef45255f03151bb3",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="mrr_at_5",
        date=("2020-05-19", "2021-04-26"),
        domains=["Written", "Non-fiction", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{möller2021germanquad,
  archiveprefix = {arXiv},
  author = {Timo Möller and Julian Risch and Malte Pietsch},
  eprint = {2104.12741},
  primaryclass = {cs.CL},
  title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
  year = {2021},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"], self.metadata_dict["eval_splits"]
        )
        self.data_loaded = True
