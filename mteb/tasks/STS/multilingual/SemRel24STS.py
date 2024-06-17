from __future__ import annotations

from mteb.abstasks import AbsTaskSTS, MultilingualTask, TaskMetadata

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


class SemRel24STS(AbsTaskSTS, MultilingualTask):
    metadata = TaskMetadata(
        name="SemRel24STS",
        dataset={
            "path": "SemRel/SemRel2024",
            "revision": "ef5c383d1b87eb8feccde3dfb7f95e42b1b050dd",
        },
        description=(
            "SemRel2024 is a collection of Semantic Textual Relatedness (STR) datasets for 14 languages, "
            "including African and Asian languages. The datasets are composed of sentence pairs, each assigned a "
            "relatedness score between 0 (completely) unrelated and 1 (maximally related) with a large range of "
            "expected relatedness values."
        ),
        reference="https://huggingface.co/datasets/SemRel/SemRel2024",
        type="STS",
        category="s2s",
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2023-01-01", "2023-12-31"),
        form=["spoken", "written"],
        domains=[],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@misc{ousidhoum2024semrel2024,
        title={SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages}, 
        author={Nedjma Ousidhoum and Shamsuddeen Hassan Muhammad and Mohamed Abdalla and Idris Abdulmumin and Ibrahim Said Ahmad and
        Sanchit Ahuja and Alham Fikri Aji and Vladimir Araujo and Abinew Ali Ayele and Pavan Baswani and Meriem Beloucif and
        Chris Biemann and Sofia Bourhim and Christine De Kock and Genet Shanko Dekebo and
        Oumaima Hourrane and Gopichand Kanumolu and Lokesh Madasu and Samuel Rutunda and Manish Shrivastava and
        Thamar Solorio and Nirmal Surange and Hailegnaw Getaneh Tilaye and Krishnapriya Vishnubhotla and Genta Winata and
        Seid Muhie Yimam and Saif M. Mohammad},
              year={2024},
              eprint={2402.08638},
              archivePrefix={arXiv},
              primaryClass={cs.CL}
        }
        """,
        n_samples={"dev": 2089, "test": 7498},
        avg_character_length={"dev": 163.1, "test": 145.9},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict

    def dataset_transform(self) -> None:
        for lang, subset in self.dataset.items():
            self.dataset[lang] = subset.rename_column("label", "score")
