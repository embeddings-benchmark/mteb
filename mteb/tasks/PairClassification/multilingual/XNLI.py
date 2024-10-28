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
        descriptive_stats={
            "n_samples": {"validation": 2163, "test": 2460},
            "test": {
                "num_samples": 19110,
                "avg_sentence1_len": 103.23793825222397,
                "avg_sentence2_len": 48.88895866038723,
                "unique_labels": 2,
                "labels": {"0": {"count": 9562}, "1": {"count": 9548}},
                "hf_subset_descriptive_stats": {
                    "ar": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 89.57362637362637,
                        "avg_sentence2_len": 41.99487179487179,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "bg": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 110.01611721611722,
                        "avg_sentence2_len": 51.62930402930403,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "de": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 119.92600732600732,
                        "avg_sentence2_len": 56.794871794871796,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "el": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 119.05421245421246,
                        "avg_sentence2_len": 56.93260073260073,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "en": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 105.67032967032966,
                        "avg_sentence2_len": 49.8043956043956,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "es": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 115.43296703296703,
                        "avg_sentence2_len": 54.68205128205128,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "fr": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 121.0967032967033,
                        "avg_sentence2_len": 58.58021978021978,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "hi": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 104.63443223443224,
                        "avg_sentence2_len": 50.17289377289377,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "ru": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 110.76923076923077,
                        "avg_sentence2_len": 52.452014652014654,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "sw": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 104.43956043956044,
                        "avg_sentence2_len": 49.48205128205128,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "th": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 96.6923076923077,
                        "avg_sentence2_len": 44.544322344322346,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "tr": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 103.67765567765568,
                        "avg_sentence2_len": 49.18534798534799,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "vi": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 111.31208791208792,
                        "avg_sentence2_len": 52.46007326007326,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "zh": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 33.03589743589744,
                        "avg_sentence2_len": 15.73040293040293,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                },
            },
            "validation": {
                "num_samples": 19110,
                "avg_sentence1_len": 103.20790162218734,
                "avg_sentence2_len": 49.01909994767138,
                "unique_labels": 2,
                "labels": {"0": {"count": 9562}, "1": {"count": 9548}},
                "hf_subset_descriptive_stats": {
                    "ar": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 88.31868131868131,
                        "avg_sentence2_len": 41.61172161172161,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "bg": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 109.196336996337,
                        "avg_sentence2_len": 51.967032967032964,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "de": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 119.81172161172161,
                        "avg_sentence2_len": 57.36923076923077,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "el": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 119.87545787545787,
                        "avg_sentence2_len": 56.88278388278388,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "en": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 105.71648351648352,
                        "avg_sentence2_len": 49.87619047619047,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "es": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 115.17289377289377,
                        "avg_sentence2_len": 55.120879120879124,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "fr": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 121.75897435897436,
                        "avg_sentence2_len": 59.08864468864469,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "hi": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 105.06446886446886,
                        "avg_sentence2_len": 50.44395604395604,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "ru": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 109.74725274725274,
                        "avg_sentence2_len": 52.26886446886447,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "sw": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 104.32234432234432,
                        "avg_sentence2_len": 49.87692307692308,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "th": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 97.28498168498169,
                        "avg_sentence2_len": 43.843223443223444,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "tr": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 102.96630036630036,
                        "avg_sentence2_len": 49.63809523809524,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "vi": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 112.26373626373626,
                        "avg_sentence2_len": 52.432967032967035,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                    "zh": {
                        "num_samples": 1365,
                        "avg_sentence1_len": 33.41098901098901,
                        "avg_sentence2_len": 15.846886446886447,
                        "unique_labels": 2,
                        "labels": {"0": {"count": 683}, "1": {"count": 682}},
                    },
                },
            },
        },
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
        descriptive_stats={
            "n_samples": {"test": 5010},
            "avg_character_length": {"test": 80.06},
        },  # average of premise and hypothesis
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
