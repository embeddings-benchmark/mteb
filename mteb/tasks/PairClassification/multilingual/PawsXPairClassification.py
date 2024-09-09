from __future__ import annotations

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PawsXPairClassification(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsXPairClassification",
        dataset={
            "path": "google-research-datasets/paws-x",
            "revision": "8a04d940a42cd40658986fdd8e3da561533a3646",
            "trust_remote_code": True,
        },
        description="{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
        reference="https://arxiv.org/abs/1908.11828",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test", "validation"],
        eval_langs={
            "de": ["deu-Latn"],
            "en": ["eng-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "ja": ["jpn-Hira"],
            "ko": ["kor-Hang"],
            "zh": ["cmn-Hans"],
        },
        main_score="max_ap",
        date=("2016-01-01", "2018-12-31"),
        domains=["Web", "Encyclopaedic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="https://huggingface.co/datasets/google-research-datasets/paws-x#licensing-information",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation="""@misc{yang2019pawsx,
      title={PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}, 
      author={Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
      year={2019},
      eprint={1908.11828},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        descriptive_stats={
            "n_samples": {"validation": 14000, "test": 14000},
            "test": {
                "num_samples": 14000,
                "avg_sentence1_len": 91.17892857142857,
                "avg_sentence2_len": 91.10121428571429,
                "unique_labels": 2,
                "labels": {"1": {"count": 6285}, "0": {"count": 7715}},
                "hf_subset_descriptive_stats": {
                    "de": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 119.7815,
                        "avg_sentence2_len": 119.2355,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 895}, "0": {"count": 1105}},
                    },
                    "en": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 113.7575,
                        "avg_sentence2_len": 113.4235,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 907}, "0": {"count": 1093}},
                    },
                    "es": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 117.815,
                        "avg_sentence2_len": 117.798,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 907}, "0": {"count": 1093}},
                    },
                    "fr": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 120.028,
                        "avg_sentence2_len": 119.9885,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 903}, "0": {"count": 1097}},
                    },
                    "ja": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 58.678,
                        "avg_sentence2_len": 58.875,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 883}, "0": {"count": 1117}},
                    },
                    "ko": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 64.9605,
                        "avg_sentence2_len": 65.114,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 896}, "0": {"count": 1104}},
                    },
                    "zh": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 43.232,
                        "avg_sentence2_len": 43.274,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 894}, "0": {"count": 1106}},
                    },
                },
            },
            "validation": {
                "num_samples": 14000,
                "avg_sentence1_len": 90.12585714285714,
                "avg_sentence2_len": 90.2045,
                "unique_labels": 2,
                "labels": {"1": {"count": 5948}, "0": {"count": 8052}},
                "hf_subset_descriptive_stats": {
                    "de": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 116.82,
                        "avg_sentence2_len": 117.0015,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 831}, "0": {"count": 1169}},
                    },
                    "en": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 113.1075,
                        "avg_sentence2_len": 112.858,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 863}, "0": {"count": 1137}},
                    },
                    "es": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 116.3285,
                        "avg_sentence2_len": 116.7275,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 847}, "0": {"count": 1153}},
                    },
                    "fr": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 119.5045,
                        "avg_sentence2_len": 119.7505,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 860}, "0": {"count": 1140}},
                    },
                    "ja": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 57.5105,
                        "avg_sentence2_len": 57.317,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 854}, "0": {"count": 1146}},
                    },
                    "ko": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 65.162,
                        "avg_sentence2_len": 65.5155,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 840}, "0": {"count": 1160}},
                    },
                    "zh": {
                        "num_samples": 2000,
                        "avg_sentence1_len": 42.448,
                        "avg_sentence2_len": 42.2615,
                        "unique_labels": 2,
                        "labels": {"1": {"count": 853}, "0": {"count": 1147}},
                    },
                },
            },
        },
    )

    def dataset_transform(self):
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            for split in self.metadata.eval_splits:
                hf_dataset = self.dataset[lang][split]

                _dataset[lang][split] = [
                    {
                        "sentence1": hf_dataset["sentence1"],
                        "sentence2": hf_dataset["sentence2"],
                        "labels": hf_dataset["label"],
                    }
                ]
        self.dataset = _dataset
