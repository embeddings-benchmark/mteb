from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class ActivityNetCaptionsT2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ActivityNetCaptionsT2VRetrieval",
        description=(
            "Retrieve the video clip that matches a given English caption from "
            "ActivityNet Captions. Each example pairs one caption with one reference "
            "video (1:1 retrieval)."
        ),
        reference="https://huggingface.co/datasets/mteb/ActivityNet_Captions_val2",
        dataset={
            "path": "mteb/ActivityNet_Captions_val2",
            "revision": "e87473c4832ea982bbeca1dde94bbebfa6ea6ada",
        },
        type="Any2AnyRetrieval",
        category="t2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Web", "Spoken"],
        task_subtypes=["Caption Pairing"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{krishna2017dense,
  author = {Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  title = {Dense-Captioning Events in Videos},
  year = {2017},
}
""",
        prompt={
            "query": "Find the video clip that matches the given caption.",
        },
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
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
