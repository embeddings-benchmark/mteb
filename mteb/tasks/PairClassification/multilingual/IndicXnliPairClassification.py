from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

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
            "split": "test",
            "trust_remote_code": True,
        },
        description="""INDICXNLI is similar to existing XNLI dataset in shape/form, but
        focusses on Indic language family.
        The train (392,702), validation (2,490), and evaluation sets (5,010) of English
        XNLI were translated from English into each of the eleven Indic languages. IndicTrans
        is a large Transformer-based sequence to sequence model. It is trained on Samanantar
        dataset (Ramesh et al., 2021), which is the largest parallel multi- lingual corpus
        over eleven Indic languages. 
        """,
        reference="https://gem-benchmark.com/data_cards/opusparcus",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ap",
        date=("2022-04-22", "2022-10-06"),
        form=["written"],
        domains=["Non-fiction", "Fiction", "Government"],
        task_subtypes=None,
        license="cc-by-4.0",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="machine-translated",
        bibtex_citation="""
        @misc{aggarwal_gupta_kunch_22,
            doi = {10.48550/ARXIV.2204.08776},
            url = {https://arxiv.org/abs/2204.08776},
            author = {Aggarwal, Divyanshu and Gupta, Vivek and Kunchukuttan, Anoop},
            title = {IndicXNLI: Evaluating Multilingual Inference for Indian Languages}, 
            publisher = {arXiv},
            year = {2022},        
            copyright = {Creative Commons Attribution 4.0 International}
        }
        """,
        n_samples={"test": 5010},
        avg_character_length={"test": 77.24},  # average of premise and hypothesis
    )

    def dataset_transform(self) -> None:
        # Convert to standard format
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            hf_dataset = self.dataset[lang]
            # 0=entailment, 2=contradiction. Filter out neutral to match the task.
            # Then map entailment as positive (1) and contradiction as negative (0).
            hf_dataset = self.dataset[lang].filter(lambda x: x["label"] in [0, 2])
            hf_dataset = hf_dataset.map(
                lambda example: {"label": 0 if example["label"] == 2 else 1}
            )
            _dataset[lang]["test"] = [
                {
                    "sentence1": hf_dataset["premise"],
                    "sentence2": hf_dataset["hypothesis"],
                    "labels": hf_dataset["label"],
                }
            ]
        self.dataset = _dataset
