import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
            + "realistic QA instances collected from literature (fiction and non-fiction) and movie scripts. "
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{kocisky-etal-2018-narrativeqa,
  address = {Cambridge, MA},
  author = {Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
Schwarz, Jonathan  and
Blunsom, Phil  and
Dyer, Chris  and
Hermann, Karl Moritz  and
Melis, G{\'a}bor  and
Grefenstette, Edward},
  doi = {10.1162/tacl_a_00023},
  editor = {Lee, Lillian  and
Johnson, Mark  and
Toutanova, Kristina  and
Roark, Brian},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {317--328},
  publisher = {MIT Press},
  title = {The {N}arrative{QA} Reading Comprehension Challenge},
  url = {https://aclanthology.org/Q18-1023},
  volume = {6},
  year = {2018},
}
""",
    )

    def load_data(self, num_proc: int = 1, **kwargs) -> None:
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            **self.metadata.dataset,
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
