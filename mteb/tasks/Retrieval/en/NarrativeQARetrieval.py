from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NarrativeQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="NarrativeQARetrieval",
        hf_hub_name="narrativeqa",
        reference="https://metatext.io/datasets/narrativeqa",
        description=(
            "NarrativeQA is a dataset for the task of question answering on long narratives. It consists of "
            "realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. "
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["en"],
        main_score="ndcg_at_10",
        revision="2e643e7363944af1c33a652d1c87320d0871c4e4",
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
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            self.metadata_dict["hf_hub_name"], split=self._EVAL_SPLIT
        )
        self.queries = {
            self._EVAL_SPLIT: {
                str(i): row["question"]["text"] for i, row in enumerate(data)
            }
        }
        self.corpus = {
            self._EVAL_SPLIT: {
                str(row["document"]["id"]): {"text": row["document"]["text"]}
                for row in data
            }
        }
        self.relevant_docs = {
            self._EVAL_SPLIT: {
                str(i): {row["document"]["id"]: 1} for i, row in enumerate(data)
            }
        }

        self.data_loaded = True
