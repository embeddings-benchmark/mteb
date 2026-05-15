from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_CONFIGS = [
    "action_antonym",
    "action_count",
    "action_localization",
    "action_prediction",
    "action_sequence",
    "character_order",
    "counterfactual_inference",
    "egocentric_navigation",
    "episodic_reasoning",
    "fine_grained_action",
    "fine_grained_pose",
    "moving_attribute",
    "moving_count",
    "moving_direction",
    "object_existence",
    "object_interaction",
    "object_shuffle",
    "scene_transition",
    "state_change",
    "unexpected_actions",
]


class MVBenchVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchVideoCentricQA",
        description="MVBench is a comprehensive video understanding benchmark covering 20 challenging video tasks, including action recognition, object interaction, scene transition, and temporal reasoning. Each example pairs a video with a question and multiple candidate answers. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset={
            "path": "mteb/MVBench",
            "revision": "69acafcca33ae9af5d6e9b6a862d6011d7a2f8dc",
        },
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-11-28", "2023-11-28"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{li2024mvbench,
  author = {Li, Kunchang and Wang, Yali and He, Yinan and Li, Yizhuo and Wang, Yi and Liu, Yi and Wang, Zun and Xu, Jilan and Chen, Guo and Luo, Ping and Wang, Limin and Qiao, Yu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title = {MVBench: A Comprehensive Multi-modal Video Understanding Benchmark},
  year = {2024},
}
""",
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            splits = []
            for cfg in _CONFIGS:
                splits.append(
                    load_dataset(
                        self.metadata.dataset["path"],
                        cfg,
                        revision=self.metadata.dataset["revision"],
                        split=split,
                    )
                )
            ds = concatenate_datasets(splits)
            ds = ds.add_column("id", [f"q{i}" for i in range(len(ds))])

            queries = ds.select_columns(["id", "question", "video"]).rename_column(
                "question", "text"
            )

            corpus_rows: list[dict] = []
            relevant_docs: dict[str, dict[str, int]] = {}
            top_ranked: dict[str, list[str]] = {}
            for row in ds.select_columns(["id", "candidates", "answer"]):
                qid = row["id"]
                answer = row["answer"]
                top_ranked[qid] = []
                for j, candidate in enumerate(row["candidates"]):
                    doc_id = f"{qid}_c{j}"
                    corpus_rows.append({"id": doc_id, "text": candidate})
                    top_ranked[qid].append(doc_id)
                    if candidate == answer:
                        relevant_docs[qid] = {doc_id: 1}

            corpus = Dataset.from_list(corpus_rows)
            self.dataset["default"][split] = RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                top_ranked=top_ranked,
            )
        self.data_loaded = True
