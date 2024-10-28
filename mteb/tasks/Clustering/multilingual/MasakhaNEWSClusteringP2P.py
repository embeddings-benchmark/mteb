from __future__ import annotations

import datasets
import numpy as np

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "amh": ["amh-Ethi"],
    "eng": ["eng-Latn"],
    "fra": ["fra-Latn"],
    "hau": ["hau-Latn"],
    "ibo": ["ibo-Latn"],
    "lin": ["lin-Latn"],
    "lug": ["lug-Latn"],
    "orm": ["orm-Ethi"],
    "pcm": ["pcm-Latn"],
    "run": ["run-Latn"],
    "sna": ["sna-Latn"],
    "som": ["som-Latn"],
    "swa": ["swa-Latn"],
    "tir": ["tir-Ethi"],
    "xho": ["xho-Latn"],
    "yor": ["yor-Latn"],
}


class MasakhaNEWSClusteringP2P(AbsTaskClustering, MultilingualTask):
    metadata = TaskMetadata(
        name="MasakhaNEWSClusteringP2P",
        description="Clustering of news article headlines and texts from MasakhaNEWS dataset. Clustering of 10 sets on the news article label.",
        reference="https://huggingface.co/datasets/masakhane/masakhanews",
        dataset={
            "path": "masakhane/masakhanews",
            "revision": "8ccc72e69e65f40c70e117d8b3c08306bb788b60",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="v_measure",
        date=(
            "2023-01-01",
            "2023-12-31",
        ),  # best guess (not found in paper, dataset or datasheet)
        domains=["News", "Written", "Non-fiction"],
        task_subtypes=["Thematic clustering"],
        license="afl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{adelani2023masakhanews,
  title={MasakhaNEWS: News Topic Classification for African languages},
  author={David Ifeoluwa Adelani and  Marek Masiak and  Israel Abebe Azime and  Jesujoba Oluwadara Alabi and  Atnafu Lambebo Tonja and  Christine Mwase and  Odunayo Ogundepo and  Bonaventure F. P. Dossou and  Akintunde Oladipo and  Doreen Nixdorf and  Chris Chinenye Emezue and  Sana Sabah al-azzawi and  Blessing K. Sibanda and  Davis David and  Lolwethu Ndolela and  Jonathan Mukiibi and  Tunde Oluwaseyi Ajayi and  Tatiana Moteu Ngoli and  Brian Odhiambo and  Abraham Toluwase Owodunni and  Nnaemeka C. Obiefuna and  Shamsuddeen Hassan Muhammad and  Saheed Salahudeen Abdullahi and  Mesay Gemeda Yigezu and  Tajuddeen Gwadabe and  Idris Abdulmumin and  Mahlet Taye Bame and  Oluwabusayo Olufunke Awoyomi and  Iyanuoluwa Shode and  Tolulope Anu Adelani and  Habiba Abdulganiy Kailani and  Abdul-Hakeem Omotayo and  Adetola Adeeko and  Afolabi Abeeb and  Anuoluwapo Aremu and  Olanrewaju Samuel and  Clemencia Siro and  Wangari Kimotho and  Onyekachi Raphael Ogbu and  Chinedu E. Mbonu and  Chiamaka I. Chukwuneke and  Samuel Fanijo and  Jessica Ojo and  Oyinkansola F. Awosan and  Tadesse Kebede Guge and  Sakayo Toadoum Sari and  Pamela Nyatsine and  Freedmore Sidume and  Oreen Yousuf and  Mardiyyah Oduwole and  Ussen Kimanuka and  Kanda Patrick Tshinu and  Thina Diko and  Siyanda Nxakama and   Abdulmejid Tuni Johar and  Sinodos Gebre and  Muhidin Mohamed and  Shafie Abdi Mohamed and  Fuad Mire Hassan and  Moges Ahmed Mehamed and  Evrard Ngabire and  and Pontus Stenetorp},
  journal={ArXiv},
  year={2023},
  volume={}
}""",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub and convert it to the standard format."""
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.hf_subsets:
            self.dataset[lang] = datasets.load_dataset(
                name=lang,
                **self.metadata_dict["dataset"],
            )
            self.dataset_transform(lang)
        self.data_loaded = True

    def dataset_transform(self, lang):
        """Convert to standard format"""
        self.dataset[lang].pop("train")
        self.dataset[lang].pop("validation")

        self.dataset[lang] = self.dataset[lang].remove_columns(
            ["url", "text", "headline"]
        )
        texts = self.dataset[lang]["test"]["headline_text"]
        labels = self.dataset[lang]["test"]["label"]
        new_format = {
            "sentences": [split.tolist() for split in np.array_split(texts, 5)],
            "labels": [split.tolist() for split in np.array_split(labels, 5)],
        }
        self.dataset[lang]["test"] = datasets.Dataset.from_dict(new_format)
