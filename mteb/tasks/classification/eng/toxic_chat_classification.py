from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_SPLITS = ["test"]


class ToxicChatClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToxicChatClassification",
        description="This dataset contains toxicity annotations on 10K user prompts collected from the Vicuna online demo. We utilize a human-AI collaborative annotation framework to guarantee the quality of annotation while maintaining a feasible annotation workload. The details of data collection, pre-processing, and annotation can be found in our paper. We believe that ToxicChat can be a valuable resource to drive further advancements toward building a safe and healthy environment for user-AI interactions. Only human annotated samples are selected here.",
        reference="https://aclanthology.org/2023.findings-emnlp.311/",
        dataset={
            "path": "lmsys/toxic-chat",
            "name": "toxicchat0124",
            "revision": "3e0319203c7162b9c9f8015b594441f979c199bc",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-26", "2024-01-31"),
        domains=["Constructed", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lin2023toxicchat,
  archiveprefix = {arXiv},
  author = {Zi Lin and Zihan Wang and Yongqi Tong and Yangkun Wang and Yuxin Guo and Yujia Wang and Jingbo Shang},
  eprint = {2310.17389},
  primaryclass = {cs.CL},
  title = {ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation},
  year = {2023},
}
""",
        superseded_by="ToxicChatClassification.v2",
    )

    def dataset_transform(self):
        keep_cols = ["user_input", "toxicity"]
        rename_dict = dict(zip(keep_cols, ["text", "label"]))
        remove_cols = [
            col
            for col in self.dataset[_EVAL_SPLITS[0]].column_names
            if col not in keep_cols
        ]
        self.dataset = self.dataset.rename_columns(rename_dict)
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
        # only use human-annotated data
        self.dataset = self.dataset.filter(lambda x: x["human_annotation"])
        self.dataset = self.dataset.remove_columns(remove_cols)


class ToxicChatClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToxicChatClassification.v2",
        description="This dataset contains toxicity annotations on 10K user prompts collected from the Vicuna online demo. We utilize a human-AI collaborative annotation framework to guarantee the quality of annotation while maintaining a feasible annotation workload. The details of data collection, pre-processing, and annotation can be found in our paper. We believe that ToxicChat can be a valuable resource to drive further advancements toward building a safe and healthy environment for user-AI interactions. Only human annotated samples are selected here. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://aclanthology.org/2023.findings-emnlp.311/",
        dataset={
            "path": "mteb/toxic_chat",
            "name": "toxicchat0124",
            "revision": "800fec53e44419d13668be291aca50a071ab5849",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=_EVAL_SPLITS,
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-10-26", "2024-01-31"),
        domains=["Constructed", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lin2023toxicchat,
  archiveprefix = {arXiv},
  author = {Zi Lin and Zihan Wang and Yongqi Tong and Yangkun Wang and Yuxin Guo and Yujia Wang and Jingbo Shang},
  eprint = {2310.17389},
  primaryclass = {cs.CL},
  title = {ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation},
  year = {2023},
}
""",
        adapted_from=["ToxicChatClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )
