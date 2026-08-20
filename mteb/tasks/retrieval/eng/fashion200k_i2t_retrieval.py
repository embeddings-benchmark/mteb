from typing import Any

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Fashion200kI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Fashion200kI2TRetrieval",
        description="Retrieve clothes based on descriptions.",
        reference="https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html",
        dataset={
            "path": "mteb/mbeir_fashion200k_task3",
            "revision": "74e9e63bf8f802f8cfb4ba6e01f48efc4f49032c",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{han2017automatic,
  author = {Han, Xintong and Wu, Zuxuan and Huang, Phoenix X and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang and Davis, Larry S},
  booktitle = {Proceedings of the IEEE international conference on computer vision},
  pages = {1463--1471},
  title = {Automatic spatially-aware fashion concept discovery},
  year = {2017},
}
""",
        prompt={
            "query": "Based on the following fashion description, retrieve the best matching image."
        },
        superseded_by="Fashion200kI2TRetrieval.v2",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # fixes https://github.com/embeddings-benchmark/mteb/issues/4436
        self.dataset["default"]["test"]["corpus"] = self.dataset["default"]["test"][
            "corpus"
        ].remove_columns("image")
        self.dataset["default"]["test"]["queries"] = self.dataset["default"]["test"][
            "queries"
        ].remove_columns("text")


class Fashion200kI2TRetrievalV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Fashion200kI2TRetrieval.v2",
        description=(
            "Retrieve clothes based on descriptions. "
            "Version 2 sets the canonical metric to hit_rate_at_10, matching the "
            "M-BEIR/UniIR source metric (hit-style Recall@10) instead of "
            "ndcg_at_10. Dataset, corpus, and qrels are identical to Fashion200kI2TRetrieval. See "
            "[Issue #5214](https://github.com/embeddings-benchmark/mteb/issues/5214)."
        ),
        reference="https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html",
        dataset={
            "path": "mteb/mbeir_fashion200k_task3",
            "revision": "74e9e63bf8f802f8cfb4ba6e01f48efc4f49032c",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="hit_rate_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{han2017automatic,
  author = {Han, Xintong and Wu, Zuxuan and Huang, Phoenix X and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang and Davis, Larry S},
  booktitle = {Proceedings of the IEEE international conference on computer vision},
  pages = {1463--1471},
  title = {Automatic spatially-aware fashion concept discovery},
  year = {2017},
}
""",
        prompt={
            "query": "Based on the following fashion description, retrieve the best matching image."
        },
        adapted_from=["Fashion200kI2TRetrieval"],
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        # fixes https://github.com/embeddings-benchmark/mteb/issues/4436
        self.dataset["default"]["test"]["corpus"] = self.dataset["default"]["test"][
            "corpus"
        ].remove_columns("image")
        self.dataset["default"]["test"]["queries"] = self.dataset["default"]["test"][
            "queries"
        ].remove_columns("text")
