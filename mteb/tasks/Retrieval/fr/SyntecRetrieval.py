from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SyntecRetrieval(AbsTaskRetrieval):
    _EVAL_SPLITS = ["test"]

    metadata = TaskMetadata(
        name="SyntecRetrieval",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p",
        hf_hub_name="lyon-nlp/mteb-fr-retrieval-syntec-s2p",
        type="Retrieval",
        category="s2p",
        eval_splits=_EVAL_SPLITS,
        eval_langs=["fr"],
        main_score="map",
        revision="b205c5084a0934ce8af14338bf03feb19499c84d",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"], "documents"
        )
        queries_raw = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"], "queries"
        )

        self.queries = {
            self._EVAL_SPLITS[0]: {
                str(i): q["Question"] for i, q in enumerate(queries_raw["queries"])
            }
        }

        corpus_raw = corpus_raw["documents"]
        corpus_raw = corpus_raw.rename_column("content", "text")
        self.corpus = {
            self._EVAL_SPLITS[0]: {str(row["id"]): row for row in corpus_raw}
        }

        self.relevant_docs = {
            self._EVAL_SPLITS[0]: {
                str(i): {str(q["Article"]): 1}
                for i, q in enumerate(queries_raw["queries"])
            }
        }

        self.data_loaded = True
