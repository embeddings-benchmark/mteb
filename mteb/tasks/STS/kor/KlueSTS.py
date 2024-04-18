from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class KlueSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KLUE-STS",
        dataset={
            "path": "klue",
            "name": "sts",
            "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
        },
        description="",
        reference="https://arxiv.org/abs/2105.09680",
        type="STS",
        category="s2s",
        eval_splits=["validation"],
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license="CC-BY-SA-4.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples={"validation": 519},
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict

    def dataset_transform(self):
        # In the case of KLUE STS, score value is nested within the `labels` field.
        # We need to extract the `score` and move it outside of the `labels` field for access.
        self.dataset["validation"] = self.dataset["validation"].map(
            lambda example: {**example, "score": example["labels"]["label"]}
        )
