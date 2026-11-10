from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset


def _transform(dataset: Dataset, lang: str | None) -> Dataset:
    dataset = dataset.rename_columns({"tweet": "text"})
    sample_size = min(2048, len(dataset["test"]))
    dataset["test"] = dataset["test"].select(range(sample_size))
    return dataset


class AfriSentiClassification(AbsTaskClassification):
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
            "swa": ["swa-Latn"],  # Swahili (macrolanguage) (Latin script)
            "twi": ["twi-Latn"],  # Twi (Latin script)
            "tso": ["tso-Latn"],  # Tsonga (Latin script)
            "yor": ["yor-Latn"],  # Yoruba (Latin script)
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
@inproceedings{muhammad_afrisenti_2023,
  title = {{A}fri{S}enti: A {T}witter Sentiment Analysis Benchmark for {A}frican Languages},
  author = {Muhammad, Shamsuddeen Hassan  and
      Abdulmumin, Idris  and
      Ayele, Abinew Ali  and
      Ousidhoum, Nedjma  and
      Adelani, David Ifeoluwa  and
      Yimam, Seid Muhie  and
      Ahmad, Ibrahim Sa'id  and
      Beloucif, Meriem  and
      Mohammad, Saif M.  and
      Ruder, Sebastian  and
      Hourrane, Oumaima  and
      Brazdil, Pavel  and
      Jorge, Alipio  and
      Ali, Felermino D{\'a}rio M{\'a}rio Ant{\'o}nio  and
      David, Davis  and
      Osei, Salomey  and
      Shehu Bello, Bello  and
      Ibrahim, Falalu  and
      Gwadabe, Tajuddeen  and
      Rutunda, Samuel  and
      Belay, Tadesse  and
      Messelle, Wendimu Baye  and
      Balcha, Hailu Beshada  and
      Chala, Sisay Adugna  and
      Gebremichael, Hagos Tesfahun  and
      Opoku, Bernard  and
      Arthur, Stephen},
  editor = {Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  month = dec,
  year = {2023},
  address = {Singapore},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2023.emnlp-main.862/},
  doi = {10.18653/v1/2023.emnlp-main.862},
  pages = {13968--13981},
}
""",
    )
