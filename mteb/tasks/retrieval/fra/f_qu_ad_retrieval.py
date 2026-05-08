import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FQuADRetrieval",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "manu/fquad2_test",
            "revision": "5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test", "validation"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2019-11-01", "2020-05-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{dhoffschmidt-etal-2020-fquad,
  address = {Online},
  author = {d{'}Hoffschmidt, Martin  and
Belblidia, Wacim  and
Heinrich, Quentin  and
Brendl{\'e}, Tom  and
Vidal, Maxime},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  doi = {10.18653/v1/2020.findings-emnlp.107},
  editor = {Cohn, Trevor  and
He, Yulan  and
Liu, Yang},
  month = nov,
  pages = {1193--1208},
  publisher = {Association for Computational Linguistics},
  title = {{FQ}u{AD}: {F}rench Question Answering Dataset},
  url = {https://aclanthology.org/2020.findings-emnlp.107},
  year = {2020},
}
""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return
        dataset_raw = datasets.load_dataset(
            **self.metadata.dataset,
        )

        # set valid_hasAns and test_hasAns as the validation and test splits (only queries with answers)
        dataset_raw["validation"] = dataset_raw["valid_hasAns"]
        del dataset_raw["valid_hasAns"]

        dataset_raw["test"] = dataset_raw["test_hasAns"]
        del dataset_raw["test_hasAns"]

        # rename  context column to text
        dataset_raw = dataset_raw.rename_column("context", "text")

        self.dataset = {}
        for eval_split in self.metadata.eval_splits:
            queries_dict = {
                str(i): q["question"] for i, q in enumerate(dataset_raw[eval_split])
            }
            corpus_dict = {
                str(row["title"]): {"title": str(row["title"]), "text": row["text"]}
                for row in dataset_raw[eval_split]
            }
            relevant_docs = {
                str(i): {str(q["title"]): 1}
                for i, q in enumerate(dataset_raw[eval_split])
            }

            corpus_dataset = Dataset.from_list(
                [
                    {"id": k, "text": v["text"], "title": v["title"]}
                    for k, v in corpus_dict.items()
                ]
            )
            queries_dataset = Dataset.from_list(
                [{"id": k, "text": v} for k, v in queries_dict.items()]
            )

            self.dataset.setdefault("default", {})[eval_split] = {
                "corpus": corpus_dataset,
                "queries": queries_dataset,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }

        self.data_loaded = True
