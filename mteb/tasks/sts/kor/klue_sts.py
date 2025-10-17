from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class KlueSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KLUE-STS",
        dataset={
            "path": "klue/klue",
            "name": "sts",
            "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
        },
        description="Human-annotated STS dataset of Korean reviews, news, and spoken word sets. Part of the Korean Language Understanding Evaluation (KLUE).",
        reference="https://arxiv.org/abs/2105.09680",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
        date=("2011-01-01", "2021-11-02"),  # rough estimate,
        domains=["Reviews", "News", "Spoken", "Written", "Spoken"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{park2021klue,
  archiveprefix = {arXiv},
  author = {Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
  eprint = {2105.09680},
  primaryclass = {cs.CL},
  title = {KLUE: Korean Language Understanding Evaluation},
  year = {2021},
}
""",
    )

    min_score = 0
    max_score = 5

    def dataset_transform(self):
        # In the case of KLUE STS, score value is nested within the `labels` field.
        # We need to extract the `score` and move it outside of the `labels` field for access.
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(
                lambda example: {**example, "score": example["labels"]["label"]}
            )
