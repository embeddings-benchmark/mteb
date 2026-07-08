from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class KorNLI(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="KorNLI",
        dataset={
            "path": "kakaobrain/kor_nli",
            "name": "xnli",
            "revision": "3e0e4626f66911b344c490c26e3cc07e6c3bb0f9",
        },
        description="Korean Natural Language Inference dataset (XNLI subset), where the "
        "Korean-translated XNLI dev/test sets are used to determine textual entailment "
        "between a premise and a hypothesis sentence. Part of KorNLI/KorSTS.",
        reference="https://arxiv.org/abs/2004.03289",
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="max_ap",
        date=("2018-01-01", "2020-04-07"),
        domains=["Non-fiction", "Fiction", "Government", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@article{ham2020kornli,
  author = {Ham, Jiyeon and Choe, Yo Joong and Park, Kyubyong and Choi, Ilji and Soh, Hyungjoon},
  journal = {arXiv preprint arXiv:2004.03289},
  title = {KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding},
  year = {2020},
}
""",
    )

    def dataset_transform(
        self,
        num_proc: int | None = None,
    ):
        _dataset = {}
        for split in self.metadata.eval_splits:
            # keep labels 0=entailment and 2=contradiction, and map them as 1 and 0 for binary classification
            hf_dataset = self.dataset[split].filter(lambda x: x["label"] in [0, 2])  # noqa: PLR6201
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
