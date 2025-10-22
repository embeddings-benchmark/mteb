import polars as pl
from datasets import concatenate_datasets, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def _load_data(path: str, splits: str, revision: str | None = None):
    corpus = {}
    queries = {}
    relevant_docs = {}

    dataset = load_dataset(
        path,
        revision=revision,
    )
    dataset_splits = ["test", "validation", "train"]
    shared_corpus = concatenate_datasets([dataset[split] for split in dataset_splits])

    text_df = pl.DataFrame({"text": shared_corpus["text"]})
    unique_indices = text_df["text"].arg_unique()
    shared_corpus = shared_corpus.select(unique_indices)

    shared_corpus = shared_corpus.map(
        lambda x: {
            "id": "corpus-" + str(x["id"]),
            "modality": "text",
        },
        remove_columns=[
            "split",
            "label",
        ],
    )

    for split in splits:
        corpus[split] = shared_corpus
        split_dataset = dataset[split]
        queries[split] = split_dataset.map(
            lambda x: {
                "id": "query-" + str(x["id"]),
                "modality": "image",
            },
            remove_columns=[
                "split",
                "label",
            ],
        )
        relevant_docs[split] = {}
        for example in split_dataset:
            query_id = "query-" + str(example["id"])
            doc_id = "corpus-" + str(example["id"])
            if query_id not in relevant_docs[split]:
                relevant_docs[split][query_id] = {}
            relevant_docs[split][query_id][doc_id] = 1

    return corpus, queries, relevant_docs


class HatefulMemesI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HatefulMemesI2TRetrieval",
        description="Retrieve captions based on memes to assess OCR abilities.",
        reference="https://arxiv.org/pdf/2005.04790",
        dataset={
            "path": "Ahren09/MMSoc_HatefulMemes",
            "revision": "c9a9a6c3ef0765622a6de0af6ebb68f323ad73ba",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@article{kiela2020hateful,
  author = {Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
  journal = {Advances in neural information processing systems},
  pages = {2611--2624},
  title = {The hateful memes challenge: Detecting hate speech in multimodal memes},
  volume = {33},
  year = {2020},
}
""",
    )

    def load_data(self) -> None:
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata.dataset["path"],
            splits=self.metadata.eval_splits,
            revision=self.metadata.dataset["revision"],
        )

        self.data_loaded = True
