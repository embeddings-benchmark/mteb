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
    dataset_splits = list(dataset)

    def map_function(split_name):
        return lambda x, idx: {
            "id": f"corpus-{split_name}-{idx}",
            "text": x["text_corrected"],
            "modality": "text",
        }

    split_datasets = {}
    for split in dataset_splits:
        split_datasets[split] = dataset[split].filter(
            lambda example: example["text_corrected"] is not None
        )

    shared_corpus = concatenate_datasets(
        [
            split_datasets[split].map(
                map_function(split),
                with_indices=True,
                remove_columns=[
                    "split",
                    "text_ocr",
                    "text_corrected",
                    "humor",
                    "sarcasm",
                    "offensive",
                    "motivational",
                    "sentiment",
                ],
            )
            for split in dataset_splits
        ]
    )

    for split in splits:
        corpus[split] = shared_corpus
        split_dataset = split_datasets[split]

        queries[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"query-{split}-{idx}",
                "modality": "image",
            },
            with_indices=True,
            remove_columns=[
                "split",
                "text_ocr",
                "humor",
                "sarcasm",
                "offensive",
                "motivational",
                "sentiment",
                "text_corrected",
            ],
        )

        relevant_docs[split] = {}
        for index in range(len(split_dataset)):
            query_id = f"query-{split}-{index}"
            doc_id = f"corpus-{split}-{index}"
            if query_id not in relevant_docs[split]:
                relevant_docs[split][query_id] = {}
            relevant_docs[split][query_id][doc_id] = 1
    return corpus, queries, relevant_docs


class MemotionI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MemotionI2TRetrieval",
        description="Retrieve captions based on memes.",
        reference="https://aclanthology.org/2020.semeval-1.99/",
        dataset={
            "path": "Ahren09/MMSoc_Memotion",
            "revision": "cdb15b61d84d56db73e0e59535dfea81ea3c22f4",
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
@inproceedings{sharma2020semeval,
  author = {Sharma, Chhavi and Bhageria, Deepesh and Scott, William and Pykl, Srinivas and Das, Amitava and Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
  booktitle = {Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  pages = {759--773},
  title = {SemEval-2020 Task 8: Memotion Analysis-the Visuo-Lingual Metaphor!},
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
