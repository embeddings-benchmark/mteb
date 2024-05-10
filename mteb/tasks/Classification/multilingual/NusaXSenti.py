from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class NusaXSentiClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NusaX-senti",
        description="NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak. NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English.",
        reference="https://arxiv.org/abs/2205.15960",
        dataset={
            "path": "indonlp/NusaX-senti",
            "revision": "a450ba4b1b6d2216c3674d3e576b2e85ce729add",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "ace": ["ace-Latn"],
            "ban": ["ban-Latn"],
            "bjn": ["bjn-Latn"],
            "bug": ["bug-Latn"],
            "eng": ["eng-Latn"],
            "ind": ["ind-Latn"],
            "jav": ["jav-Latn"],
            "mad": ["mad-Latn"],
            "min": ["min-Latn"],
            "nij": ["nij-Latn"],
            "sun": ["sun-Latn"],
            "bbc": ["bbc-Latn"],
        },
        main_score="accuracy",
        date=("2022-05-01", "2023-05-08"),
        form=["written"],
        domains=["Reviews","Web", "Social", "Constructed",],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC-BY-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{winata2022nusax,
      title={NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
      author={Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya,
      Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony,
      Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo,
      Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau,
      Jey Han and Sennrich, Rico and Ruder, Sebastian},
      year={2022},
      eprint={2205.15960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
""",
        n_samples={"test": 500},
        avg_character_length={"test": 52.4},
    )

    def dataset_transform(self):
        for lang in self.langs:
            self.dataset[lang]["test"]
