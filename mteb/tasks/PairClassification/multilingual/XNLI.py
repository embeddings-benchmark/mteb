from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification

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
            "path": "xnli",
            "revision": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
        },
        description="",
        reference="https://aclanthology.org/D18-1269/",
        category="s2s",
        type="PairClassification",
        eval_splits=["test", "validation"],
        eval_langs=_LANGS,
        main_score="ap",
        date=("2018-01-01", "2018-11-04"),
        form=["written"],
        domains=["Non-fiction", "Fiction", "Government"],
        task_subtypes=None,  # does not match any.
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation=None,
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
        n_samples=None,
        avg_character_length=None,
    )

    def dataset_transform(self):
        _dataset = {}
        for lang in self.langs:
            _dataset[lang] = {}
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
                        "sent1": hf_dataset["premise"],
                        "sent2": hf_dataset["hypothesis"],
                        "labels": hf_dataset["label"],
                    }
                ]
        self.dataset = _dataset
