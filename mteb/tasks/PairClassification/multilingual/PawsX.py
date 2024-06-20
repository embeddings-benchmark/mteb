from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PawsX(MultilingualTask, AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PawsX",
        dataset={
            "path": "google-research-datasets/paws-x",
            "revision": "8a04d940a42cd40658986fdd8e3da561533a3646",
        },
        description="",
        reference="https://arxiv.org/abs/1908.11828",
        category="s2s",
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
        main_score="ap",
        date=("2016-01-01", "2018-12-31"),
        form=["written"],
        domains=["Web", "Encyclopaedic"],
        task_subtypes=["Textual Entailment"],
        license="Custom (commercial)",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="human-translated",
        bibtex_citation="""@misc{yang2019pawsx,
      title={PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}, 
      author={Yinfei Yang and Yuan Zhang and Chris Tar and Jason Baldridge},
      year={2019},
      eprint={1908.11828},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples={"validation": 14000, "test": 14000},
        avg_character_length={"validation": 91.2, "test": 91.1},
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
