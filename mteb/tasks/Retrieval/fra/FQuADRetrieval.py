from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FQuADRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["test", "validation"]

    metadata = TaskMetadata(
        name="FQuADRetrieval",
        description="This dataset has been built from the French SQuad dataset.",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "manu/fquad2_test",
            "revision": "5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        dataset_raw = datasets.load_dataset(
            **self.metadata_dict["dataset"],
        )

        # set valid_hasAns and test_hasAns as the validation and test splits (only queries with answers)
        dataset_raw["validation"] = dataset_raw["valid_hasAns"]
        del dataset_raw["valid_hasAns"]

        dataset_raw["test"] = dataset_raw["test_hasAns"]
        del dataset_raw["test_hasAns"]

        # rename  context column to text
        dataset_raw = dataset_raw.rename_column("context", "text")

        self.queries = {
            eval_split: {
                str(i): q["question"] for i, q in enumerate(dataset_raw[eval_split])
            }
            for eval_split in self.metadata_dict["eval_splits"]
        }

        self.corpus = {
            eval_split: {str(row["title"]): row for row in dataset_raw[eval_split]}
            for eval_split in self.metadata_dict["eval_splits"]
        }

        self.relevant_docs = {
            eval_split: {
                str(i): {str(q["title"]): 1}
                for i, q in enumerate(dataset_raw[eval_split])
            }
            for eval_split in self.metadata_dict["eval_splits"]
        }

        self.data_loaded = True
