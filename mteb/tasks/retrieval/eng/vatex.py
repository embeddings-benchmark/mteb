from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class VATEXV2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VATEXV2T",
        description=(
            "VATEX is a large-scale multilingual video description dataset built on "
            "Kinetics-600 clips. The task is video-to-text retrieval: given a video "
            "query, retrieve the matching English caption from the corpus."
        ),
        dataset={
            "path": "mteb/VATEX_test_1k",
            "revision": "0d2e86e6d36927f4676ee6127c4e38e3867ce0ce",
        },
        type="Any2AnyRetrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://huggingface.co/datasets/mteb/VATEX_test_1k",
        category="v2t",
        modalities=["video", "text"],
        date=("2019-01-01", "2019-12-31"),
        domains=["Web"],
        task_subtypes=["Activity recognition"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{wang2019vatex,
  title={{VATEX}: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research},
  author={Wang, Xin and Wu, Jiawei and Chen, Junkun and Li, Lei and Wang, Yuan-Fang and Wang, William Yang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}""",
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            split=self.metadata.eval_splits[0],
        )
        dataset = dataset.add_column("id", [str(i) for i in range(len(dataset))])
        query = dataset.select_columns(["id", "video"])
        corpus = dataset.select_columns(["id", "caption"]).rename_column(
            "caption", "text"
        )
        qrels = {str(i): {str(i): 1} for i in range(len(dataset))}

        self.dataset["default"]["test"] = RetrievalSplitData(
            queries=query, corpus=corpus, relevant_docs=qrels, top_ranked=None
        )
        self.data_loaded = True
