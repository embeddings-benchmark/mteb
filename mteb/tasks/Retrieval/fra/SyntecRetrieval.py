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
        dataset={
            "path": "lyon-nlp/mteb-fr-retrieval-syntec-s2p",
            "revision": "19661ccdca4dfc2d15122d776b61685f48c68ca9",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=None,  # not specified
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{ciancone2024extending,
  archiveprefix = {arXiv},
  author = {Mathieu Ciancone and Imene Kerboua and Marion Schaeffer and Wissam Siblini},
  eprint = {2405.20468},
  primaryclass = {cs.CL},
  title = {Extending the Massive Text Embedding Benchmark to French},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(
            name="documents",
            **self.metadata_dict["dataset"],
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata_dict["dataset"],
        )

        eval_split = self.metadata_dict["eval_splits"][0]
        self.queries = {
            eval_split: {
                str(i): q["Question"] for i, q in enumerate(queries_raw[eval_split])
            }
        }

        corpus_raw = corpus_raw[eval_split]
        corpus_raw = corpus_raw.rename_column("content", "text")
        self.corpus = {eval_split: {str(row["id"]): row for row in corpus_raw}}

        self.relevant_docs = {
            eval_split: {
                str(i): {str(q["Article"]): 1}
                for i, q in enumerate(queries_raw[eval_split])
            }
        }

        self.data_loaded = True
