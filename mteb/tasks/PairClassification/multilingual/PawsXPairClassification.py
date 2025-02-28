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
