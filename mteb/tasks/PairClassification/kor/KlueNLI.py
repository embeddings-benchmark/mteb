from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class KlueNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="KLUE-NLI",
        dataset={
            "path": "klue/klue",
            "name": "nli",
            "revision": "349481ec73fff722f88e0453ca05c77a447d967c",
        },
        description="Textual Entailment between a hypothesis sentence and a premise sentence. Part of the Korean Language Understanding Evaluation (KLUE).",
        reference="https://arxiv.org/abs/2105.09680",
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["kor-Hang"],
        main_score="max_ap",
        date=("2016-01-01", "2020-12-31"),
        domains=["News", "Encyclopaedic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",  # 3000 - neutral samples
        descriptive_stats={
            "n_samples": {"validation": 2000},
            "avg_character_length": {"validation": 35.01},
        },
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            # keep labels 0=entailment and 2=contradiction, and map them as 1 and 0 for binary classification
            hf_dataset = self.dataset[split].filter(lambda x: x["label"] in [0, 2])
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 0 if example["label"] == 2 else 1}
            )
            _dataset[split] = [
                {
                    "sentence1": hf_dataset["premise"],
                    "sentence2": hf_dataset["hypothesis"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
