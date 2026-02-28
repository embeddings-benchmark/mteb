import datasets

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


def _transform(dataset, lang):
    dataset = dataset.rename_columns({"tweet": "text"})

    # Handle languages without train split (like Oromo)
    if "train" not in dataset and "dev" in dataset:
        # Use dev split as train split for languages that don't have train
        dataset["train"] = dataset["dev"]

    sample_size = min(2048, len(dataset["test"]))
    dataset["test"] = dataset["test"].select(range(sample_size))
    return dataset


class AfriSentiClassification(AbsTaskClassification):
    fast_loading = True
    metadata = TaskMetadata(
        name="AfriSentiClassification",
        description="AfriSenti is the largest sentiment analysis dataset for under-represented African languages.",
        dataset={
            "path": "mteb/AfriSentiClassification",
            "revision": "463850b9da81847227ff03f3f80a3310b93cba3c",
        },
        reference="https://arxiv.org/abs/2302.08956",
        type="Classification",
        category="t2c",
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
            "swh": ["swa-Latn"],  # Swahili (macrolanguage) (Latin script)
            "twi": ["twi-Latn"],  # Twi (Latin script)
            "tso": ["tso-Latn"],  # Tsonga (Latin script)
            "yor": ["yor-Latn"],  # Yoruba (Latin script)
            "gaz": ["orm-Ethi"],
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
            dataset_metadata = self.metadata.dataset
            try:
                dataset = datasets.load_dataset(name=lang, **dataset_metadata)
            except Exception as e:
                # Some languages (e.g., Oromo) do not provide a train split in AfriSenti.
                # When the upstream dataset script expects a train split, HF may raise:
                #   ValueError: Instruction "train" corresponds to no data!
                # In that case, load available splits and map validation -> train.
                msg = str(e)
                if 'Instruction "train" corresponds to no data' in msg:
                    dev = datasets.load_dataset(
                        name=lang, split="validation", **dataset_metadata
                    )
                    test = datasets.load_dataset(
                        name=lang, split="test", **dataset_metadata
                    )
                    dataset = datasets.DatasetDict({"train": dev, "test": test})
                else:
                    raise
            self.dataset[lang] = _transform(dataset, lang)
        self.data_loaded = True
