from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, MultilingualTask

_LANGUAGES = {
    "afr": ["afr-Latn"],
    "arq": ["arq-Arab"],
    "amh": ["amh-Ethi"],
    "eng": ["eng-Latn"],
    "hau": ["hau-Latn"],
    "ind": ["ind-Latn"],
    "hin": ["hin-Deva"],
    "kin": ["kin-Latn"],
    "mar": ["mar-Deva"],
    "arb": ["arb-Arab"],
    "ary": ["ary-Arab"],
    "pan": ["pan-Guru"],
    # gold test scores are hidden
    # "esp": ["esp"],
    "tel": ["tel-Telu"],
}


_SPLITS = ["dev", "test"]


class STSSemRel2024(AbsTaskSTS, MultilingualTask):
    metadata = TaskMetadata(
        name="STSSemRel2024",
        dataset={
            "path": "SemRel/SemRel2024",
            "revision": "ef5c383d1b87eb8feccde3dfb7f95e42b1b050dd",
        },
        description=("Semantic Textual Relatedness for African and Asian Languages"),
        reference="https://semantic-textual-relatedness.github.io/",
        type="STS",
        category="s2s",
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=["News", "Non-fiction", "Web", "Spoken"],
        task_subtypes=[],
        license="CC0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{ousidhoum2024semrel2024,
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
        n_samples={"dev": 2471, "test": 8732},
        avg_character_length={"dev": 166.48, "test": 151.28},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        def get_dataset_subset(lang: str):
            """For a specified subset (=language)
            only get the splits listed in _SPLIT
            and rename column "score"

            Args:
                lang (str): _description_

            Returns:
                datasets.DatasetDict: the dataset of the specified language
            """
            subset = datasets.DatasetDict(
                **dict(
                    zip(
                        _SPLITS,
                        datasets.load_dataset(
                            name=lang,
                            split=_SPLITS,
                            **self.metadata_dict["dataset"],
                        ),
                    )
                )
            )
            return subset.rename_column("label", "score")

        self.dataset = datasets.DatasetDict(
            **dict(zip(self.langs, [get_dataset_subset(lang) for lang in self.langs]))
        )
        self.data_loaded = True
