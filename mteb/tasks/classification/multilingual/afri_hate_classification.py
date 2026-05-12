from __future__ import annotations

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AfriHateClassification(AbsTaskClassification):
    """
    AfriHate: A Multilingual African Hate Speech and Abusive Language Dataset.
    Each sample is a tweet annotated by native speakers with sociocultural understanding
    of the context and language, addressing the crucial need for localized and
    community-driven moderation resources.
    """

    metadata = TaskMetadata(
        name="AfriHateClassification",
        description=(
            "AfriHate is a multilingual collection of hate speech and abusive language "
            "datasets covering 15 African languages. Each example is a tweet annotated "
            "by native speakers with sociocultural understanding of the context and language, "
            "addressing the crucial need for localized and community-driven moderation resources."
        ),
        reference="https://aclanthology.org/2025.naacl-long.92/",
        dataset={
            "path": "mteb/AfriHateClassification",
            "revision": "ec8740ea28c08bfeb9cd99b7ded40fb73582297b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            # (ISO-639-3 code : BCP-47 tag)
            "arq": ["arq-Arab"],  # Algerian Arabic
            "amh": ["amh-Ethi"],  # Amharic
            "ibo": ["ibo-Latn"],  # Igbo
            "kin": ["kin-Latn"],  # Kinyarwanda
            "hau": ["hau-Latn"],  # Hausa
            "ary": ["ary-Arab"],  # Moroccan Arabic
            "pcm": ["pcm-Latn"],  # Nigerian Pidgin
            "orm": ["gaz-Latn"],  # Oromo
            "som": ["som-Latn"],  # Somali
            "swa": ["swh-Latn"],  # Swahili
            "tir": ["tir-Ethi"],  # Tigrinya
            "twi": ["twi-Latn"],  # Twi
            "xho": ["xho-Latn"],  # isiXhosa
            "yor": ["yor-Latn"],  # Yorùbá
            "zul": ["zul-Latn"],  # isiZulu
        },
        main_score="accuracy",
        date=("2025-01-01", "2025-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""@inproceedings{muhammad-etal-2025-afrihate,
  address = {Albuquerque, New Mexico},
  author = {Muhammad, Shamsuddeen Hassan  and
Abdulmumin, Idris  and
Ayele, Abinew Ali  and
Adelani, David Ifeoluwa  and
Ahmad, Ibrahim Said  and
Aliyu, Saminu Mohammad  and
R{\"o}ttger, Paul  and
Oppong, Abigail  and
Bukula, Andiswa  and
Chukwuneke, Chiamaka Ijeoma  and
Jibril, Ebrahim Chekol  and
Ismail, Elyas Abdi  and
Alemneh, Esubalew  and
Gebremichael, Hagos Tesfahun  and
Aliyu, Lukman Jibril  and
Beloucif, Meriem  and
Hourrane, Oumaima  and
Mabuya, Rooweither  and
Osei, Salomey  and
Rutunda, Samuel  and
Belay, Tadesse Destaw  and
Guge, Tadesse Kebede  and
Asfaw, Tesfa Tegegne  and
Wanzare, Lilian Diana Awuor  and
Onyango, Nelson Odhiambo  and
Yimam, Seid Muhie  and
Ousidhoum, Nedjma},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  editor = {Chiruzzo, Luis  and
Ritter, Alan  and
Wang, Lu},
  isbn = {979-8-89176-189-6},
  month = apr,
  pages = {1854--1871},
  publisher = {Association for Computational Linguistics},
  title = {{A}fri{H}ate: A Multilingual Collection of Hate Speech and Abusive Language Datasets for {A}frican Languages},
  url = {https://aclanthology.org/2025.naacl-long.92/},
  year = {2025},
}
""",
    )

    def dataset_transform(self, **kwargs) -> None:
        """
        Transform the dataset to MTEB expected format:

        * column **text**: tweet content (str)
        * column **label**: int (0=normal, 1=abusive, 2=hate)
        """
        for lang in self.dataset:
            for split in self.dataset[lang]:
                ds = self.dataset[lang][split]

                # Map label strings to integers
                label_map = {"Normal": 0, "Abuse": 1, "Hate": 2}

                def transform_labels(example):
                    return {
                        "text": example["tweet"],
                        "label": label_map[example["label"]],
                    }

                ds = ds.map(
                    transform_labels,
                    remove_columns=ds.column_names,
                    desc=f"{lang}/{split}",
                )

                self.dataset[lang][split] = ds
