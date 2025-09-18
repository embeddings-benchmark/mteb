from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class GqnliPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="GqnliPairClassification",
        description=(
            "Natural Language Inference on GQNLI: "
            "predict the relation between two sentences "
            "(0=implication, 1=neutral, 2=contradiction)."
        ),
        reference="https://huggingface.co/datasets/maximoss/gqnli-fr",
        dataset={
            "path": "maximoss/gqnli-fr",
            "revision": "3089f0b591628692e7c4d4122804f6995380bdad",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="max_accuracy",
        date=("2024-01-01", "2024-12-31"),
        domains=["Academic"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""@inproceedings{skandalis-etal-2024-new-datasets,
    title = "New Datasets for Automatic Detection of Textual Entailment and of Contradictions between Sentences in {F}rench",
    author = "Skandalis, Maximos  and
      Moot, Richard  and
      Retor{\'e}, Christian  and
      Robillard, Simon",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1065",
    pages = "12173--12186",
}
""",
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
                        "labels": hf_dataset["label"],
                    }
                ]
        self.dataset = _dataset
