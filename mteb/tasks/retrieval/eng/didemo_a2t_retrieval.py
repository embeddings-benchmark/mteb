from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class DiDeMoA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DiDeMoA2TRetrieval",
        description=(
            "Retrieve the English caption that describes a given audio track "
            "from the DiDeMo dataset of Flickr videos with temporally grounded "
            "sentence descriptions."
        ),
        dataset={
            "path": "mteb/DiDeMo",
            "revision": "746689f644b66022540a9a39136e842bee164e6b",
        },
        type="Any2AnyRetrieval",
        category="a2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        reference="https://arxiv.org/abs/1708.01641",
        modalities=["audio", "text"],
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{hendricks2017localizing,
  author = {Hendricks, Lisa Anne and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  title = {Localizing Moments in Video with Natural Language},
  year = {2017},
}
""",
        prompt={
            "query": "Find the caption that describes the following audio.",
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load the DiDeMo dataset for audio-to-text retrieval.

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

        query = dataset.select_columns(["id", "audio"])
        corpus = dataset.select_columns(["id", "caption"]).rename_column(
            "caption", "text"
        )
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
