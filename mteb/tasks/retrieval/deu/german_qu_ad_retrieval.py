from collections import defaultdict

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_retrieval_data(dataset_path, eval_splits):
    eval_split = eval_splits[0]
    corpus_dataset = load_dataset(dataset_path, "corpus")
    queries_dataset = load_dataset(dataset_path, "queries")
    qrels = load_dataset(dataset_path + "-qrels")[eval_split]

    corpus_dict = {
        e["_id"]: {"text": e["text"], "title": ""} for e in corpus_dataset["corpus"]
    }
    queries_dict = {e["_id"]: e["text"] for e in queries_dataset["queries"]}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e["query-id"]][e["corpus-id"]] = e["score"]

    corpus_ds = Dataset.from_list(
        [
            {"id": k, "text": v["text"], "title": v["title"]}
            for k, v in corpus_dict.items()
        ]
    )
    queries_ds = Dataset.from_list(
        [{"id": k, "text": v} for k, v in queries_dict.items()]
    )

    return {
        "default": {
            eval_split: {
                "corpus": corpus_ds,
                "queries": queries_ds,
                "relevant_docs": dict(relevant_docs),
                "top_ranked": None,
            }
        }
    }


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
        category="t2t",
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

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = load_retrieval_data(
            self.metadata.dataset["path"], self.metadata.eval_splits
        )
        self.data_loaded = True
