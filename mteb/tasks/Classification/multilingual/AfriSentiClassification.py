from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


def _transform(dataset, lang):
    dataset = dataset.rename_columns({"tweet": "text"})
    
    # Handle languages without train split (like Oromo)
    if "train" not in dataset and "dev" in dataset:
        # Use dev split as train split for languages that don't have train
        dataset["train"] = dataset["dev"]
    
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
            "trust_remote_code": True,
        },
        reference="https://arxiv.org/abs/2302.08956",
        type="Classification",
        category="s2s",
        modalities=["text"],
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
            "orm": ["orm-Ethi"],
        },
        main_score="accuracy",
        date=("2023-02-16", "2023-09-03"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Muhammad2023AfriSentiAT,
  author = {Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Abinew Ali Ayele and Nedjma Ousidhoum and David Ifeoluwa Adelani and Seid Muhie Yimam and Ibrahim Sa'id Ahmad and Meriem Beloucif and Saif Mohammad and Sebastian Ruder and Oumaima Hourrane and Pavel Brazdil and Felermino D'ario M'ario Ant'onio Ali and Davis Davis and Salomey Osei and Bello Shehu Bello and Falalu Ibrahim and Tajuddeen Gwadabe and Samuel Rutunda and Tadesse Belay and Wendimu Baye Messelle and Hailu Beshada Balcha and Sisay Adugna Chala and Hagos Tesfahun Gebremichael and Bernard Opoku and Steven Arthur},
  title = {AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages},
  year = {2023},
}
""",
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.hf_subsets:
            metadata = self.metadata_dict.get("dataset", None)
            try:
                dataset = datasets.load_dataset(name=lang, **metadata)
            except Exception as e:
                # Some languages (e.g., Oromo) do not provide a train split in AfriSenti.
                # When the upstream dataset script expects a train split, HF may raise:
                #   ValueError: Instruction "train" corresponds to no data!
                # In that case, load available splits and map validation -> train.
                msg = str(e)
                if "Instruction \"train\" corresponds to no data" in msg:
                    dev = datasets.load_dataset(name=lang, split="validation", **metadata)
                    test = datasets.load_dataset(name=lang, split="test", **metadata)
                    dataset = datasets.DatasetDict({"train": dev, "test": test})
                else:
                    raise
            self.dataset[lang] = _transform(dataset, lang)
        self.dataset_transform()
        self.data_loaded = True
