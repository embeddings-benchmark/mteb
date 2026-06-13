from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@inproceedings{li2024mvbench,
  author = {Li, Kunchang and Wang, Yali and He, Yinan and Li, Yizhuo and Wang, Yi and Liu, Yi and Wang, Zun and Xu, Jilan and Chen, Guo and Luo, Ping and Wang, Limin and Qiao, Yu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  title = {MVBench: A Comprehensive Multi-modal Video Understanding Benchmark},
  year = {2024},
}
"""

_DATASET = {
    "path": "mteb/MVBench",
    "revision": "69acafcca33ae9af5d6e9b6a862d6011d7a2f8dc",
}

_DATE = ("2023-11-28", "2023-11-28")


def _load_split(config, split):
    ds = load_dataset(
        _DATASET["path"], config, revision=_DATASET["revision"], split=split
    )
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
    return RetrievalSplitData(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        top_ranked=top_ranked,
    )


class MVBenchActionAntonymVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchActionAntonymVideoCentricQA",
        description="MVBench Action Antonym subset: questions about identifying the opposite of an observed action in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("action_antonym", split)
        self.data_loaded = True


class MVBenchActionCountVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchActionCountVideoCentricQA",
        description="MVBench Action Count subset: questions about counting the number of times an action occurs in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("action_count", split)
        self.data_loaded = True


class MVBenchActionLocalizationVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchActionLocalizationVideoCentricQA",
        description="MVBench Action Localization subset: questions about localizing when a specific action occurs in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("action_localization", split)
        self.data_loaded = True


class MVBenchActionPredictionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchActionPredictionVideoCentricQA",
        description="MVBench Action Prediction subset: questions about predicting the next action based on the video context. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("action_prediction", split)
        self.data_loaded = True


class MVBenchActionSequenceVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchActionSequenceVideoCentricQA",
        description="MVBench Action Sequence subset: questions about the temporal order of actions in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("action_sequence", split)
        self.data_loaded = True


class MVBenchCharacterOrderVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchCharacterOrderVideoCentricQA",
        description="MVBench Character Order subset: questions about the order in which characters appear in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("character_order", split)
        self.data_loaded = True


class MVBenchCounterfactualInferenceVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchCounterfactualInferenceVideoCentricQA",
        description="MVBench Counterfactual Inference subset: questions requiring counterfactual reasoning about what would happen differently in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("counterfactual_inference", split)
        self.data_loaded = True


class MVBenchEgocentricNavigationVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchEgocentricNavigationVideoCentricQA",
        description="MVBench Egocentric Navigation subset: questions about navigation and spatial understanding from an egocentric video perspective. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("egocentric_navigation", split)
        self.data_loaded = True


class MVBenchEpisodicReasoningVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchEpisodicReasoningVideoCentricQA",
        description="MVBench Episodic Reasoning subset: questions requiring memory and reasoning over events across the full video episode. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("episodic_reasoning", split)
        self.data_loaded = True


class MVBenchFineGrainedActionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchFineGrainedActionVideoCentricQA",
        description="MVBench Fine-Grained Action subset: questions distinguishing between visually similar fine-grained actions in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("fine_grained_action", split)
        self.data_loaded = True


class MVBenchFineGrainedPoseVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchFineGrainedPoseVideoCentricQA",
        description="MVBench Fine-Grained Pose subset: questions about identifying fine-grained human body poses and gestures in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("fine_grained_pose", split)
        self.data_loaded = True


class MVBenchMovingAttributeVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchMovingAttributeVideoCentricQA",
        description="MVBench Moving Attribute subset: questions about the attributes of moving objects in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("moving_attribute", split)
        self.data_loaded = True


class MVBenchMovingCountVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchMovingCountVideoCentricQA",
        description="MVBench Moving Count subset: questions about counting moving objects in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("moving_count", split)
        self.data_loaded = True


class MVBenchMovingDirectionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchMovingDirectionVideoCentricQA",
        description="MVBench Moving Direction subset: questions about the direction of movement of objects in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("moving_direction", split)
        self.data_loaded = True


class MVBenchObjectExistenceVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchObjectExistenceVideoCentricQA",
        description="MVBench Object Existence subset: questions about whether a specific object appears in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("object_existence", split)
        self.data_loaded = True


class MVBenchObjectInteractionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchObjectInteractionVideoCentricQA",
        description="MVBench Object Interaction subset: questions about how objects interact with each other in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("object_interaction", split)
        self.data_loaded = True


class MVBenchObjectShuffleVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchObjectShuffleVideoCentricQA",
        description="MVBench Object Shuffle subset: questions about tracking objects after they have been shuffled or rearranged in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("object_shuffle", split)
        self.data_loaded = True


class MVBenchSceneTransitionVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchSceneTransitionVideoCentricQA",
        description="MVBench Scene Transition subset: questions about detecting and understanding scene transitions in the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("scene_transition", split)
        self.data_loaded = True


class MVBenchStateChangeVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchStateChangeVideoCentricQA",
        description="MVBench State Change subset: questions about how the state of objects or scenes changes throughout the video. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("state_change", split)
        self.data_loaded = True


class MVBenchUnexpectedActionsVideoCentricQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MVBenchUnexpectedActionsVideoCentricQA",
        description="MVBench Unexpected Actions subset: questions about identifying actions that are surprising or unexpected given the video context. The task is formulated as multiple-choice retrieval: given the (video, question) pair, retrieve the correct candidate.",
        reference="https://arxiv.org/abs/2311.17005",
        dataset=_DATASET,
        type="VideoCentricQA",
        category="vt2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=_DATE,
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["video", "text"],
        sample_creation="found",
        is_beta=True,
        bibtex_citation=_BIBTEX,
    )

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        self.dataset = {"default": {}}
        for split in self.metadata.eval_splits:
            self.dataset["default"][split] = _load_split("unexpected_actions", split)
        self.data_loaded = True
