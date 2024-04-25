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
    # gold labels are hidden
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
        description=(
            "A Collection of Semantic Textual Relatedness Datasets for 14 Languages"
        ),
        reference="https://semantic-textual-relatedness.github.io",
        type="STS",
        category="s2s",
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation=None,
        citation="""@inproceedings{ousidhoum-etal-2024-semeval,
'title': '{S}em{E}val-2024 Task 1: Semantic Textual Relatedness for African and Asian Languages',
'author': 'Ousidhoum, Nedjma and Muhammad, Shamsuddeen Hassan and Abdalla, Mohamed and Abdulmumin, Idris and
Ahmad,Ibrahim Said and Ahuja, Sanchit and Aji, Alham Fikri and Araujo, Vladimir and     Beloucif, Meriem and
De Kock, Christine and Hourrane, Oumaima and Shrivastava, Manish and Solorio, Thamar and Surange, Nirmal and
Vishnubhotla, Krishnapriya and Yimam, Seid Muhie and Mohammad, Saif M.',
'booktitle': 'Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024)',
'year': '2024',
'publisher': 'Association for Computational Linguistics'
}""",
        n_samples=None,
        avg_character_length=None,
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
