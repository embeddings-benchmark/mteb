from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _load_data(path: str, splits: str, cache_dir: str = None, revision: str = None):
    corpus = {}
    queries = {}
    relevant_docs = {}

    dataset = load_dataset(
        path,
        cache_dir=cache_dir,
        revision=revision,
    )
    dataset_splits = list(dataset)

    def map_function(split_name):
        return lambda x, idx: {
            "id": f"corpus-{split_name}-{idx}",
            "text": None,
            "modality": "image",
        }

    # Apply the map function to each split and concatenate
    shared_corpus = concatenate_datasets(
        [
            dataset[split].map(
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
    # image corrupted
    shared_corpus = shared_corpus.select(
        [i for i in range(len(shared_corpus)) if i not in [4578, 6781, 6784, 6786]]
    )
    for split in splits:
        corpus[split] = shared_corpus
        split_dataset = dataset[split]
        queries[split] = split_dataset.map(
            lambda x, idx: {
                "id": f"query-{split}-{idx}",
                "text": x["text_corrected"],
                "modality": "text",
                "image": None,
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
        if split == "test":
            queries[split] = queries[split].select(
                [i for i in range(len(queries[split])) if i not in [489, 492, 494]]
            )
        relevant_docs[split] = {}
        for index in range(len(split_dataset)):
            if index not in [489, 492, 494]:
                query_id = f"query-{split}-{index}"
                doc_id = f"corpus-{split}-{index}"
                if query_id not in relevant_docs[split]:
                    relevant_docs[split][query_id] = {}
                relevant_docs[split][query_id][doc_id] = 1
    return corpus, queries, relevant_docs


class MemotionT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="MemotionT2IRetrieval",
        description="Retrieve memes based on captions.",
        reference="https://aclanthology.org/2020.semeval-1.99/",
        dataset={
            "path": "Ahren09/MMSoc_Memotion",
            "revision": "cdb15b61d84d56db73e0e59535dfea81ea3c22f4",
            # "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="t2i",
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
        bibtex_citation="""@inproceedings{sharma2020semeval,
  title={SemEval-2020 Task 8: Memotion Analysis-the Visuo-Lingual Metaphor!},
  author={Sharma, Chhavi and Bhageria, Deepesh and Scott, William and Pykl, Srinivas and Das, Amitava and Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
  booktitle={Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  pages={759--773},
  year={2020}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1.0,
                    "average_query_length": 83.80057388809182,
                    "num_documents": 6988,
                    "num_queries": 697,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )

        self.data_loaded = True
