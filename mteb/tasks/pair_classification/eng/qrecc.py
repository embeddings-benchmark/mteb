from typing import Any

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class QRECC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="QRECC",
        description="QRECC.",
        reference=None,
        dataset={
            "path": "DeepPavlov/qrecc",
            "revision": "449692c06b7d01d1d089037700c3dd495fb36278",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{anantha2021opendomainquestionansweringgoes,
      title={Open-Domain Question Answering Goes Conversational via Question Rewriting}, 
      author={Raviteja Anantha and Svitlana Vakulenko and Zhucheng Tu and Shayne Longpre and Stephen Pulman and Srinivas Chappidi},
      year={2021},
      eprint={2010.04898},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2010.04898}, 
}
""",
    )

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        def transform(example: dict) -> dict:
            context_str = ""
            for replic in example["context"]:
                if replic["role"] == "user":
                    context_str += "User: " + replic["content"] + " "
                else:
                    context_str += "Assistant: " + replic["content"] + " "
            context_str += example["question"]
            return {
                "sentence1": context_str,
                "sentence2": example["rewrite"],
                "labels": 1,
            }

        self.dataset = self.dataset.map(transform, num_proc=num_proc)
