import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SyntecRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SyntecRetrieval",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-retrieval-syntec-s2p",
        dataset={
            "path": "lyon-nlp/mteb-fr-retrieval-syntec-s2p",
            "revision": "19661ccdca4dfc2d15122d776b61685f48c68ca9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),  # publication year
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

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return
        # fetch both subsets of the dataset
        corpus_raw = datasets.load_dataset(
            name="documents",
            **self.metadata.dataset,
        )
        queries_raw = datasets.load_dataset(
            name="queries",
            **self.metadata.dataset,
        )

        eval_split = self.metadata.eval_splits[0]
        queries_dict = {
            str(i): q["Question"] for i, q in enumerate(queries_raw[eval_split])
        }

        corpus_split = corpus_raw[eval_split]
        corpus_split = corpus_split.rename_column("content", "text")
        corpus_dict = {
            str(row["id"]): {"title": "", "text": row["text"]} for row in corpus_split
        }

        relevant_docs = {
            str(i): {str(q["Article"]): 1}
            for i, q in enumerate(queries_raw[eval_split])
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

        self.dataset = {
            "default": {
                eval_split: {
                    "corpus": corpus_dataset,
                    "queries": queries_dataset,
                    "relevant_docs": relevant_docs,
                    "top_ranked": None,
                }
            }
        }

        self.data_loaded = True
