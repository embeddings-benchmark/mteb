from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FQuADRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["test_hasAns", "valid_hasAns"]

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
        date=None,
        form=None,
        domains=["Encyclopaedic"],
        task_subtypes=None,
        license="apache-2.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{faysse2024croissantllm,
                      title={CroissantLLM: A Truly Bilingual French-English Language Model}, 
                      author={Manuel Faysse and Patrick Fernandes and Nuno M. Guerreiro and António Loison and Duarte M. Alves and Caio Corro and Nicolas Boizard and João Alves and Ricardo Rei and Pedro H. Martins and Antoni Bigata Casademunt and François Yvon and André F. T. Martins and Gautier Viaud and Céline Hudelot and Pierre Colombo},
                      year={2024},
                      eprint={2402.00786},
                      archivePrefix={arXiv},
                      primaryClass={cs.CL}
                }""",
        n_samples={"test": 400, "validation": 100},
        avg_character_length={"test": 700, "validation": 700},
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        dataset_raw = datasets.load_dataset(
            **self.metadata_dict["dataset"],
        )

        # rename  context column to text
        dataset_raw = dataset_raw.rename_column("context", "text")

        self.queries = {
            eval_split: {
                str(i): q["question"] for i, q in enumerate(dataset_raw[eval_split])
            } for eval_split in self.metadata_dict["eval_splits"]
        }

        self.corpus = {eval_split: {str(row["title"]): row for row in dataset_raw[eval_split]}
                       for eval_split in self.metadata_dict["eval_splits"]
                       }

        self.relevant_docs = {
            eval_split: {
                str(i): {str(q["title"]): 1}
                for i, q in enumerate(dataset_raw[eval_split])
            } for eval_split in self.metadata_dict["eval_splits"]
        }

        self.data_loaded = True
