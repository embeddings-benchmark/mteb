from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MSRVTTV2T(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSRVTTV2T",
        description="A large video description dataset for bridging video and language",
        dataset={
            "path": "mteb/MSR-VTT",
            "revision": "4661603cee25c1fd370e5478a2953203cf37155b",
        },
        type="Retrieval",
        eval_langs=["eng-Latn"],
        eval_splits=["test"],
        main_score="ndcg_at_10",
        reference="https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf",
        category="va2t",
        modalities=["audio", "video", "text"],
        date=("2016-01-01", "2016-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{xu2016msrvtt,
  author = {Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {Msr-vtt: A large video description dataset for bridging video and language},
  year = {2016},
}
""",
        is_beta=True,
    )

    input_column_name = ("video", "audio")

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load the MSRVTT dataset.

        TODO: Reupload dataset in standard format and remove this custom load_data.
        """
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
