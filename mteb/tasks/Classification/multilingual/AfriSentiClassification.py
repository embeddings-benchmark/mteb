from __future__ import annotations

import datasets

from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


def _transform(dataset, lang):
    dataset = dataset.rename_columns({"tweet": "text"})
    sample_size = min(2048, len(dataset["test"]))
    dataset["test"] = dataset["test"].select(range(sample_size))
    return dataset


class AfriSentiClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="AfriSentiClassification",
        description="AfriSenti is the largest sentiment analysis dataset for under-represented African languages.",
        dataset={
            "path": "shmuhammad/AfriSenti-twitter-sentiment",
            "revision": "b52e930385cf5ed7f063072c3f7bd17b599a16cf",
        },
        reference="https://arxiv.org/abs/2302.08956",
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "amh": ["amh-Ethi"],  # Amharic (Ethiopic script)
            "arq": ["arq-Arab"],
            "ary": ["ary-Arab"],  # Moroccan Arabic, Standard Arabic (Arabic script)
            "hau": [
                "hau-Latn"
            ],  # Hausa (Latin script), additional script if written in Ajami (Arabic script)
            "ibo": ["ibo-Latn"],  # Igbo (Latin script)
            "kin": ["kin-Latn"],  # Kinyarwanda (Latin script)
            "por": ["por-Latn"],  # Portuguese (Latin script)
            "pcm": ["pcm-Latn"],  # Nigerian Pidgin (Latin script)
            "swa": ["swa-Latn"],  # Swahili (macrolanguage) (Latin script)
            "twi": ["twi-Latn"],  # Twi (Latin script)
            "tso": ["tso-Latn"],  # Tsonga (Latin script)
            "yor": ["yor-Latn"],  # Yoruba (Latin script)
        },
        main_score="accuracy",
        date=("2023-02-16", "2023-09-03"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="Creative Commons Attribution 4.0 International License",
        socioeconomic_status="low",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{Muhammad2023AfriSentiAT,
        title=AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages,
        author=Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Abinew Ali Ayele and Nedjma Ousidhoum and David Ifeoluwa Adelani and Seid Muhie Yimam and Ibrahim Sa'id Ahmad and Meriem Beloucif and Saif Mohammad and Sebastian Ruder and Oumaima Hourrane and Pavel Brazdil and Felermino D'ario M'ario Ant'onio Ali and Davis Davis and Salomey Osei and Bello Shehu Bello and Falalu Ibrahim and Tajuddeen Gwadabe and Samuel Rutunda and Tadesse Belay and Wendimu Baye Messelle and Hailu Beshada Balcha and Sisay Adugna Chala and Hagos Tesfahun Gebremichael and Bernard Opoku and Steven Arthur,
        year=2023
        }""",
        n_samples={"test": 2048},
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            metadata = self.metadata_dict.get("dataset", None)
            dataset = datasets.load_dataset(name=lang, **metadata)
            self.dataset[lang] = _transform(dataset, lang)
        self.dataset_transform()
        self.data_loaded = True
