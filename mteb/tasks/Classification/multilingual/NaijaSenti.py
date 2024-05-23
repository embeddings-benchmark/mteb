from typing import Any

import datasets
from mteb.abstasks import AbsTaskClassification, MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class NaijaSenti(AbsTaskClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="NaijaSenti",
        description="NaijaSenti is the first large-scale human-annotated Twitter sentiment dataset for the four most widely spoken languages in Nigeria — Hausa, Igbo, Nigerian-Pidgin, and Yorùbá — consisting of around 30,000 annotated tweets per language, including a significant fraction of code-mixed tweets.",
        reference="https://github.com/hausanlp/NaijaSenti",
        dataset={
            "path": "HausaNLP/NaijaSenti-Twitter",
            "revision": "a3d0415a828178edf3466246f49cfcd83b946ab3",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs={
            "hau": ["hau-Latn"],
            "ibo": ["ibo-Latn"],
            "pcm": ["pcm-Latn"],
            "yor": ["yor-Latn"],
        },
        main_score="accuracy",
        date=("2022-05-01", "2023-05-08"),
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="CC-BY-4.0 license",
        socioeconomic_status="low",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @inproceedings{muhammad-etal-2022-naijasenti,
    title = "{N}aija{S}enti: A {N}igerian {T}witter Sentiment Corpus for Multilingual Sentiment Analysis",
    author = "Muhammad, Shamsuddeen Hassan  and
      Adelani, David Ifeoluwa  and
      Ruder, Sebastian  and
      Ahmad, Ibrahim Sa{'}id  and
      Abdulmumin, Idris  and
      Bello, Bello Shehu  and
      Choudhury, Monojit  and
      Emezue, Chris Chinenye  and
      Abdullahi, Saheed Salahudeen  and
      Aremu, Anuoluwapo  and
      Jorge, Al{\'\i}pio  and
      Brazdil, Pavel",
        booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
        month = jun,
        year = "2022",
        address = "Marseille, France",
        publisher = "European Language Resources Association",
        url = "https://aclanthology.org/2022.lrec-1.63",
        pages = "590--602",
    }""",
        n_samples={"test": 4800},
        avg_character_length={"test": 72.81},
    )

    def load_data(self, **kwargs: Any) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=f"{lang}",
                **self.metadata_dict["dataset"],
            )
            self.dataset[lang] = datasets.DatasetDict(
                {
                    "train": self.dataset[lang]["train"],
                    "test": self.dataset[lang]["test"],
                }
            )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        for lang in self.hf_subsets:
            self.dataset[lang] = self.dataset[lang].map(
                lambda example: {"text": example["tweet"]}
            )
