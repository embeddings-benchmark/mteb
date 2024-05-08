from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class XNLITurkishV2(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XNLITurkishV2",
        dataset={
            "path": "Harsit/xnli2.0_turkish",
            "revision": "dde1ce992ca9090e9b36466c98d017ffad2b294c",
        },
        description="",
        reference="https://arxiv.org/abs/2301.06527",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ap",
        date=("2012-01-01", "2023-01-01"),
        form=["written"],
        domains=["Non-fiction", "Fiction", "Government"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="machine-translated and verified",
        bibtex_citation="""@inproceedings{upadhyay2023xnli,
  title={XNLI 2.0: Improving XNLI dataset and performance on Cross Lingual Understanding (XLU)},
  author={Upadhyay, Ankit Kumar and Upadhya, Harsit Kumar},
  booktitle={2023 IEEE 8th International Conference for Convergence in Technology (I2CT)},
  pages={1--6},
  year={2023},
  organization={IEEE}
        """,
        n_samples={"test": 5010},
        avg_character_length={"test": 77.51},
    )

    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]

            # 0=entailment, 2=contradiction. Filter out neutral to match the task.
            # Then map entailment as positive (1) and contradiction as negative (0).
            hf_dataset = hf_dataset.filter(lambda x: x["label"] in [0, 2])
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 0 if example["label"] == 2 else 1}
            )

            _dataset[split] = [
                {
                    "sent1": hf_dataset["premise"],
                    "sent2": hf_dataset["hypothesis"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
