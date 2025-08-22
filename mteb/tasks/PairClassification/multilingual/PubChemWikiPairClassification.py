from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "de": ["deu-Latn", "eng-Latn"],
    "nl": ["nld-Latn", "eng-Latn"],
    "zh": ["zho-Hans", "eng-Latn"],
    "fr": ["fra-Latn", "eng-Latn"],
    "es": ["spa-Latn", "eng-Latn"],
    "pt": ["por-Latn", "eng-Latn"],
    "ms": ["msa-Latn", "eng-Latn"],
    "ko": ["kor-Hang", "eng-Latn"],
    "tr": ["tur-Latn", "eng-Latn"],
    "hi": ["hin-Deva", "eng-Latn"],
    "cs": ["ces-Latn", "eng-Latn"],
    "ja": ["jpn-Jpan", "eng-Latn"],
}


class PubChemWikiPairClassification(AbsTaskPairClassification, MultilingualTask):
    metadata = TaskMetadata(
        name="PubChemWikiPairClassification",
        dataset={
            "path": "BASF-AI/PubChemWikiMultilingualPC",
            "revision": "3412b208896a37e4ebb5ff7b96f6cc313ee9d2e3",
        },
        description="ChemTEB evaluates the performance of text embedding models on chemical domain data.",
        reference="https://arxiv.org/abs/2412.00532",
        category="s2s",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="max_ap",
        date=("2024-06-01", "2024-11-30"),
        domains=["Chemistry"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{kasmaee2024chemteb,
  author = {Kasmaee, Ali Shiraee and Khodadad, Mohammad and Saloot, Mohammad Arshi and Sherck, Nick and Dokas, Stephen and Mahyar, Hamidreza and Samiee, Soheila},
  journal = {arXiv preprint arXiv:2412.00532},
  title = {ChemTEB: Chemical Text Embedding Benchmark, an Overview of Embedding Models Performance \\& Efficiency on a Specific Domain},
  year = {2024},
}

@article{kim2023pubchem,
  author = {Kim, Sunghwan and Chen, Jie and Cheng, Tiejun and Gindulyte, Asta and He, Jia and He, Siqian and Li, Qingliang and Shoemaker, Benjamin A and Thiessen, Paul A and Yu, Bo and others},
  journal = {Nucleic acids research},
  number = {D1},
  pages = {D1373--D1380},
  publisher = {Oxford University Press},
  title = {PubChem 2023 update},
  volume = {51},
  year = {2023},
}
""",
    )

    def dataset_transform(self) -> None:
        _dataset = {}
        for lang in self.hf_subsets:
            _dataset[lang] = {}
            hf_dataset = self.dataset[lang][self.metadata.eval_splits[0]]
            _dataset[lang]["test"] = [
                {
                    "sentence1": hf_dataset["sent1"],
                    "sentence2": hf_dataset["sent2"],
                    "labels": hf_dataset["labels"],
                }
            ]
        self.dataset = _dataset
