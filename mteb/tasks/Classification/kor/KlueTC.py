from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KlueTC(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KLUE-TC",
        dataset={
            "path": "klue",
            "name": "ynat",
            "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
        },
        description="Topic classification dataset of human-annotated news headlines. Part of the Korean Language Understanding Evaluation (KLUE).",
        reference="https://arxiv.org/abs/2105.09680",
        type="Classification",
        category="s2s",
        eval_splits=["validation"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
        date=("2016-01-01", "2020-12-31"),  # from 2016 to 2020
        form=["written"],
        domains=["News"],
        task_subtypes=["Topic classification"],
        license="CC-BY-SA-4.0",
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples={"validation": 2048},
        avg_character_length={"validation": 27.079609091907326},
    )

    def dataset_transform(self):
        def id2str(example):
            return {"label": label_feature.int2str(example["label_id"])}

        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["validation"]
        )

        label_feature = self.dataset[self.metadata.eval_splits[0]].features["label"]

        self.dataset = self.dataset.rename_columns(
            {"title": "text", "label": "label_id"}
        )
        self.dataset = self.dataset.map(id2str)
