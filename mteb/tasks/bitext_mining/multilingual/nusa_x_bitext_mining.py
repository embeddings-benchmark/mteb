from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_LANGUAGES = {
    "eng-ace": ["eng-Latn", "ace-Latn"],
    "eng-ban": ["eng-Latn", "ban-Latn"],
    "eng-bbc": ["eng-Latn", "bbc-Latn"],
    "eng-bjn": ["eng-Latn", "bjn-Latn"],
    "eng-bug": ["eng-Latn", "bug-Latn"],
    "eng-ind": ["eng-Latn", "ind-Latn"],
    "eng-jav": ["eng-Latn", "jav-Latn"],
    "eng-mad": ["eng-Latn", "mad-Latn"],
    "eng-min": ["eng-Latn", "min-Latn"],
    "eng-nij": ["eng-Latn", "nij-Latn"],
    "eng-sun": ["eng-Latn", "sun-Latn"],
}


class NusaXBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="NusaXBitextMining",
        dataset={
            "path": "mteb/NusaXBitextMining",
            "revision": "6db681bd37d6827f8d132590429ce33a2af4c96e",
        },
        description="NusaX is a parallel dataset for machine translation and sentiment analysis on 11 Indonesia languages and English.",
        reference="https://huggingface.co/datasets/indonlp/NusaX-senti/",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{winata2023nusax,
  author = {Winata, Genta Indra and Aji, Alham Fikri and Cahyawijaya, Samuel and Mahendra, Rahmad and Koto, Fajri and Romadhony, Ade and Kurniawan, Kemal and Moeljadi, David and Prasojo, Radityo Eko and Fung, Pascale and others},
  booktitle = {Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics},
  pages = {815--834},
  title = {NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages},
  year = {2023},
}

@inproceedings{winata2024miners,
  address = {Miami, Florida, USA},
  author = {Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2024},
  doi = {10.18653/v1/2024.findings-emnlp.155},
  editor = {Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung},
  month = {nov},
  pages = {2742--2766},
  publisher = {Association for Computational Linguistics},
  title = {{MINERS}: Multilingual Language Models as Semantic Retrievers},
  url = {https://aclanthology.org/2024.findings-emnlp.155/},
  year = {2024},
}
""",
    )
