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
        eval_splits=_EVAL_SPLITS,
        eval_langs=["fra-Latn"],
        main_score="map",
        date=("2019-11-01", "2020-05-01"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{dhoffschmidt2020fquad,
      title={FQuAD: French Question Answering Dataset}, 
      author={Martin d'Hoffschmidt and Wacim Belblidia and Tom Brendlé and Quentin Heinrich and Maxime Vidal},
      year={2020},
      eprint={2002.06071},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples={"test": 400, "validation": 100},
        avg_character_length={"test": 937, "validation": 930},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        dataset_raw = datasets.load_dataset(
            **self.metadata_dict["dataset"],
        )

        # set valid_hasAns as the validation split
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
