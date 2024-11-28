from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "ar": ["ara-Arab"],
    "bg": ["bul-Cyrl"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "hi": ["hin-Deva"],
    "ru": ["rus-Cyrl"],
    "sw": ["swa-Latn"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "vi": ["vie-Latn"],
    "zh": ["zho-Hans"],
}


class XNLI(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XNLI",
        dataset={
            "path": "mteb/xnli",
            "revision": "09698e0180d87dc247ca447d3a1248b931ac0cdb",
        },
        description="",
        reference="https://aclanthology.org/D18-1269/",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test", "validation"],
        eval_langs=_LANGS,
        main_score="max_ap",
        date=("2018-01-01", "2018-11-04"),
        domains=["Non-fiction", "Fiction", "Government", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@InProceedings{conneau2018xnli,
        author = {Conneau, Alexis
                        and Rinott, Ruty
                        and Lample, Guillaume
                        and Williams, Adina
                        and Bowman, Samuel R.
                        and Schwenk, Holger
                        and Stoyanov, Veselin},
        title = {XNLI: Evaluating Cross-lingual Sentence Representations},
        booktitle = {Proceedings of the 2018 Conference on Empirical Methods
                    in Natural Language Processing},
        year = {2018},
        publisher = {Association for Computational Linguistics},
        location = {Brussels, Belgium},
        }
        """,
    )

    def dataset_transform(self):
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang], seed=self.seed, splits=self.metadata.eval_splits
            )
            for split in self.metadata.eval_splits:
                # 0=entailment, 2=contradiction. Filter out neutral to match the task.
                # Then map entailment as positive (1) and contradiction as negative (0).
                hf_dataset = self.dataset[lang][split].filter(
                    lambda x: x["label"] in [0, 2]
                )
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


_LANGS_2 = {
    "punjabi": ["pan-Guru"],
    "gujrati": ["guj-Gujr"],
    "kannada": ["kan-Knda"],
    "assamese": ["asm-Beng"],
    "bengali": ["ben-Beng"],
    "marathi": ["mar-Deva"],
    "bhojpuri": ["bho-Deva"],
    "odiya": ["ory-Orya"],
    "sanskrit": ["san-Deva"],
    "tamil": ["tam-Taml"],
    "turkish": ["tur-Latn"],
    "greek": ["ell-Grek"],
    "russian": ["rus-Cyrl"],
}


class XNLIV2(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XNLIV2",
        dataset={
            "path": "mteb/xnli2.0-multi-pair",
            "revision": "5b7d477a8c62cdd18e2fed7e015497c20b4371ad",
        },
        description="""
        This is subset of 'XNLI 2.0: Improving XNLI dataset and performance on Cross Lingual Understanding'
        with languages that were not part of the original XNLI plus three (verified) languages that are not strongly covered in MTEB
        """,
        reference="https://arxiv.org/pdf/2301.06527",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGS_2,
        main_score="max_ap",
        date=("2018-01-01", "2018-11-04"),
        domains=["Non-fiction", "Fiction", "Government", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="""@inproceedings{upadhyay2023xnli,
            title={XNLI 2.0: Improving XNLI dataset and performance on Cross Lingual Understanding (XLU)},
            author={Upadhyay, Ankit Kumar and Upadhya, Harsit Kumar},
            booktitle={2023 IEEE 8th International Conference for Convergence in Technology (I2CT)},
            pages={1--6},
            year={2023},
            organization={IEEE}
            }
        """,
        # average of premise and hypothesis
    )

    def dataset_transform(self):
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang], seed=self.seed, splits=self.metadata.eval_splits
            )
            for split in self.metadata.eval_splits:
                # 0=entailment, 2=contradiction. Filter out neutral to match the task.
                # Then map entailment as positive (1) and contradiction as negative (0).
                hf_dataset = self.dataset[lang][split].filter(
                    lambda x: x["label"] in [0, 2]
                )
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
