from datasets import load_dataset
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GqnliTask(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="gqnli",
        description=(
            "Natural Language Inference on GQNLI: "
            "predict the relation between two sentences "
            "(0=implication, 1=neutral, 2=contradiction)."
        ),
        reference="https://huggingface.co/datasets/maximoss/gqnli-fr",
        dataset={
            "path": "maximoss/gqnli-fr",
            "revision": "main"
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="max_accuracy",
        date=("2025-08-05", "2025-08-05"),
        domains=["Academic"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@article{your_citation_here,
  title={Your Title Here},
  author={Your Author Here},
  journal={Your Journal Here},
  year={2025}
}""",
    )

    def dataset_transform(self):
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            for split in self.metadata.eval_splits:
                hf_dataset = self.dataset[split]
                hf_dataset = hf_dataset.filter(lambda x: x["label"] in [0, 2])
                hf_dataset = hf_dataset.map(
                    lambda example: {"label": 0 if example["label"] == 2 else 1}
                )
                _dataset[lang][split] = [
                    {
                        "sentence1": hf_dataset["premise"],
                        "sentence2": hf_dataset["hypothesis"],
                        "labels":    hf_dataset["label"],
                    }
                ]
        self.dataset = _dataset