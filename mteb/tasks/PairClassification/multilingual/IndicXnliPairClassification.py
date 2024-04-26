from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskPairClassification, MultilingualTask

_LANGUAGES = {
    "as": ["asm-Beng"],
    "bn": ["ben-Beng"],
    "gu": ["guj-Gujr"],
    "hi": ["hin-Deva"],
    "kn": ["kan-Knda"],
    "ml": ["mal-Mlym"],
    "mr": ["mar-Deva"],
    "or": ["ory-Orya"],
    "pa": ["pan-Guru"],
    "ta": ["tam-Taml"],
    "te": ["tel-Telu"],
}




class IndicXnliPairClassification(AbsTaskPairClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="IndicXnliPairClassification",
        dataset={
            "path": "Divyanshu/indicxnli",
            "revision": "7092c27872e919f31d0496fb8b9c47bd2cba3f6c",
        },
        description="INDICXNLI is similar to existing XNLI dataset in shape/form, but focusses on Indic language family",
        reference="https://gem-benchmark.com/data_cards/opusparcus",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    def dataset_transform(self) -> None:
        # Convert to standard format
        for lang in self.langs:
            self.dataset[lang] = self.dataset[lang].rename_columns(
                {"premise": "sent1", "hypothesis": "sent2", "label": "labels"}
            )
            self.dataset[lang] = self.stratified_subsampling(
                self.dataset[lang], n_samples=256, seed=self.seed, label="labels"
            )
            print(self.dataset)
