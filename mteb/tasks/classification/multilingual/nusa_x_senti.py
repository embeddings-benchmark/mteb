from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NusaXSentiClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NusaX-senti",
        description="NusaX is a high-quality multilingual parallel corpus that covers 12 languages, Indonesian, English, and 10 Indonesian local languages, namely Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak. NusaX-Senti is a 3-labels (positive, neutral, negative) sentiment analysis dataset for 10 Indonesian local languages + Indonesian and English.",
        reference="https://arxiv.org/abs/2205.15960",
        dataset={
            "path": "mteb/NusaX-senti",
            "revision": "7b2ff4facde8a6473f667a38edef1d03ce9ca0dc",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
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
        domains=["Reviews", "Web", "Social", "Constructed", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{winata2022nusax,
  archiveprefix = {arXiv},
  author = {Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya,
Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony,
Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo,
Radityo Eko and Fung, Pascale and Baldwin, Timothy and Lau,
Jey Han and Sennrich, Rico and Ruder, Sebastian},
  eprint = {2205.15960},
  primaryclass = {cs.CL},
  title = {NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
  year = {2022},
}
""",
    )
