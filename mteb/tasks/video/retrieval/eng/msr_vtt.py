from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MSRVTTV2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTV2T",
        description="MSRVTT",
        dataset={
            "path": "mteb/MSR-VTT",
            "revision": "4661603cee25c1fd370e5478a2953203cf37155b",
        },
        type="Retrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference=None,
        category="v2t",
        modalities=["video", "text"],
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load the MSRVTT dataset."""
        if self.data_loaded:
            return
        dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split=self.metadata.eval_splits[0],
        )
        dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])
        query = dataset.select_columns(["id", "video", "audio"])
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


if __name__ == "__main__":
    import logging

    import mteb

    logging.basicConfig(level=logging.INFO)

    task = MSRVTTV2T()
    model = mteb.get_model("baseline/random-encoder-baseline")
    mteb.evaluate(model, task)
