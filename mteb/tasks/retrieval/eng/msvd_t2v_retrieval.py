from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MSVDT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSVDT2VRetrieval",
        description=(
            "Retrieve the short video clip that matches a given English caption. "
            "Each example pairs one caption with one video (1:1 retrieval)."
        ),
        reference="https://huggingface.co/datasets/mteb/MSVD",
        dataset={
            "path": "mteb/MSVD",
            "revision": "177cefe5957363d1617f0427fcfa00ff0b0d7da8",
        },
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2010-01-01", "2012-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["text", "video"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{microsoft2011msvd,
  author = {{Microsoft Research}},
  title = {Microsoft Research Video Description Corpus},
  url = {https://www.microsoft.com/en-us/research/publication/microsoft-research-video-description-corpus/},
  year = {2011},
}
""",
        prompt={
            "query": "Find the video clip that matches the given caption.",
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load the MSVD dataset for text-to-video retrieval.

        TODO: Reupload dataset in standard format and remove this custom load_data.
        """
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split=self.metadata.eval_splits[0],
        )
        dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])

        query = dataset.select_columns(["id", "caption"]).rename_column(
            "caption", "text"
        )
        corpus = dataset.select_columns(["id", "video"])
        qrels = {}
        for i in range(len(dataset)):
            key = str(i)
            if key not in qrels:
                qrels[key] = {}
            qrels[key][key] = 1
        self.dataset["default"]["test"] = RetrievalSplitData(
            queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
        )
        self.data_loaded = True
