from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "afr": ["afr-Latn"],
    "amh": ["amh-Ethi"],
    "arb": ["arb-Arab"],
    "arq": ["arq-Arab"],
    "ary": ["ary-Arab"],
    "eng": ["eng-Latn"],
    "hau": ["hau-Latn"],
    "hin": ["hin-Deva"],
    "ind": ["ind-Latn"],
    "kin": ["kin-Latn"],
    "mar": ["mar-Deva"],
    "tel": ["tel-Telu"],
}

_SPLITS = ["test"]


class SemRel24STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SemRel24STS",
        dataset={
            "path": "mteb/SemRel24STS",
            "revision": "f5146bf724c55899ca20502851dc98c5405b67f9",
        },
        description=(
            "SemRel2024 is a collection of Semantic Textual Relatedness (STR) datasets for 14 languages, "
            "including African and Asian languages. The datasets are composed of sentence pairs, each assigned a "
            "relatedness score between 0 (completely) unrelated and 1 (maximally related) with a large range of "
            "expected relatedness values."
        ),
        reference="https://huggingface.co/datasets/SemRel/SemRel2024",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2023-01-01", "2023-12-31"),
        domains=["Spoken", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{ousidhoum2024semrel2024,
  author = {Ousidhoum, Nedjma and Muhammad, Shamsuddeen and Abdalla, Mohamed and Abdulmumin, Idris and Ahmad, Ibrahim and Ahuja, Sanchit and Aji, Alham and Araujo, Vladimir and Ayele, Abinew and Baswani, Pavan and Beloucif, Meriem and Biemann, Chris and Bourhim, Sofia and Kock, Christine and Dekebo, Genet and Hourrane, Oumaima and Kanumolu, Gopichand and Madasu, Lokesh and Rutunda, Samuel and Shrivastava, Manish and Solorio, Thamar and Surange, Nirmal and Tilaye, Hailegnaw and Vishnubhotla, Krishnapriya and Winata, Genta and Yimam, Seid and Mohammad, Saif},
  booktitle = {Findings of the Association for Computational Linguistics ACL 2024},
  doi = {10.18653/v1/2024.findings-acl.147},
  pages = {2512–2530},
  publisher = {Association for Computational Linguistics},
  title = {SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 13 Languages},
  url = {https://doi.org/10.18653/v1/2024.findings-acl.147},
  year = {2024},
}
""",
    )

    min_score = 0
    max_score = 1
