from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NarrativeQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="NarrativeQARetrieval",
        dataset={
            "path": "deepmind/narrativeqa",
            "revision": "2e643e7363944af1c33a652d1c87320d0871c4e4",
        },
        reference="https://metatext.io/datasets/narrativeqa",
        description=(
            "NarrativeQA is a dataset for the task of question answering on long narratives. It consists of "
            "realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. "
        ),
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{kočiský2017narrativeqa,
      title={The NarrativeQA Reading Comprehension Challenge}, 
      author={Tomáš Kočiský and Jonathan Schwarz and Phil Blunsom and Chris Dyer and Karl Moritz Hermann and Gábor Melis and Edward Grefenstette},
      year={2017},
      eprint={1712.07040},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples=None,
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
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
