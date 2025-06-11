from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class KlueMrcDomainClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="KlueMrcDomainClustering",
        description="this dataset is a processed and redistributed version of the KLUE-MRC dataset. Domain: Game / Media / Automotive / Finance / Real Estate / Education",
        reference="https://huggingface.co/datasets/on-and-on/clustering_klue_mrc_context_domain",
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="v_measure",
        dataset={
            "path": "on-and-on/clustering_klue_mrc_context_domain",
            "revision": "a814b5ef0b6814991785f2c31af8e38ef7bb3f0d",
        },
        date=("2016-01-01", "2020-12-31"),
        domains=["News", "Written"],
        task_subtypes=[],
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
        prompt="Identify the topic or theme of the given texts",
    )

    def dataset_transform(self):
        documents: list = []
        labels: list = []

        split = self.metadata.eval_splits[0]
        ds = {}

        self.dataset = self.dataset.rename_columns(
            {"text": "sentences", "label": "labels"}
        )

        documents.append(self.dataset[split]["sentences"])
        labels.append(self.dataset[split]["labels"])

        ds[split] = datasets.Dataset.from_dict(
            {
                "sentences": documents,
                "labels": labels,
            }
        )
        self.dataset = datasets.DatasetDict(ds)
