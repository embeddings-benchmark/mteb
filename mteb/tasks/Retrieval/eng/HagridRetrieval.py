from __future__ import annotations

import uuid

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class HagridRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HagridRetrieval",
        dataset={
            "path": "miracl/hagrid",
            "revision": "b2a085913606be3c4f2f1a8bff1810e38bade8fa",
            "trust_remote_code": True,
        },
        reference="https://github.com/project-miracl/hagrid",
        description=(
            "HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)"
            + "is a dataset for generative information-seeking scenarios. It consists of queries"
            + "along with a set of manually labelled relevant passages"
        ),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-02-01", "2022-10-18"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{hagrid,
      title={{HAGRID}: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution},
      author={Ehsan Kamalloo and Aref Jafari and Xinyu Zhang and Nandan Thakur and Jimmy Lin},
      year={2023},
      journal={arXiv:2307.16883},
}""",
    )

    def load_data(self, **kwargs):
        """Loads the different split of the dataset (queries/corpus/relevants)"""
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            "miracl/hagrid",
            split=self.metadata.eval_splits[0],
            revision=self.metadata_dict["dataset"].get("revision", None),
            trust_remote_code=self.metadata_dict["dataset"].get(
                "trust_remote_code", False
            ),
        )
        proc_data = self.preprocess_data(data)

        self.queries = {
            self.metadata.eval_splits[0]: {
                d["query_id"]: d["query_text"] for d in proc_data
            }
        }
        self.corpus = {
            self.metadata.eval_splits[0]: {
                d["answer_id"]: {"text": d["answer_text"]} for d in proc_data
            }
        }
        self.relevant_docs = {
            self.metadata.eval_splits[0]: {
                d["query_id"]: {d["answer_id"]: 1} for d in proc_data
            }
        }

        self.data_loaded = True

    def preprocess_data(self, dataset: dict) -> list[dict]:
        """Preprocessed the data in a format easirer
        to handle for the loading of queries and corpus
        ------
        PARAMS
        dataset : the hagrid dataset (json)
        """
        preprocessed_data = []
        for d in dataset:
            # get the best answer among positively rated answers
            best_answer = self.get_best_answer(d)
            # if no good answer found, skip
            if best_answer is not None:
                preprocessed_data.append(
                    {
                        "query_id": str(d["query_id"]),
                        "query_text": d["query"],
                        "answer_id": str(uuid.uuid4()),
                        "answer_text": best_answer,
                    }
                )

        return preprocessed_data

    def get_best_answer(self, data: dict) -> str:
        """Get the best answer among available answers
        of a query.
        WARNING : May return None if no good answer available
        --------
        PARAMS:
        data: a dict representing one element of the dataset
        """
        good_answers = [
            a["answer"]
            for a in data["answers"]
            if a["informative"] == 1 and a["attributable"] == 1
        ]
        # Return 1st one if >=1 good answers else None
        return good_answers[0] if len(good_answers) > 0 else None
