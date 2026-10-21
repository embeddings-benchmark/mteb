from __future__ import annotations

from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class InjongoIntent(AbsTaskClassification):
    """Intent-classification component of the INJONGO benchmark
    (40 intents × 5 domains, 16 African languages + English)."""

    metadata = TaskMetadata(
        name="InjongoIntent",
        description=(
            "Multicultural intent-classification dataset covering banking, home, kitchen & dining, "
            "travel and utility. 3 200 utterances per African language (2 240 / 320 / 640 train/dev/test) "
            "+ 1 779 English. From ‘INJONGO: A Multicultural Intent Detection and Slot-filling "
            "Dataset for 16 African Languages’ (Yu et al., 2025)."
        ),
        reference="https://aclanthology.org/2025.acl-long.464/",
        dataset={
            "path": "mteb/InjongoIntent",
            # exact commit that matches the paper; bump when the HF repo updates
            "revision": "5af1657c7b42aecfe924c0d9a870ac621a589785",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "amh": ["amh-Ethi"],
            "ewe": ["ewe-Latn"],
            "hau": ["hau-Latn"],
            "ibo": ["ibo-Latn"],
            "kin": ["kin-Latn"],
            "lin": ["lin-Latn"],
            "lug": ["lug-Latn"],
            "gaz": ["orm-Ethi"],
            "sna": ["sna-Latn"],
            "sot": ["sot-Latn"],
            "swh": ["swa-Latn"],
            "twi": ["twi-Latn"],
            "wol": ["wol-Latn"],
            "xho": ["xho-Latn"],
            "yor": ["yor-Latn"],
            "zul": ["zul-Latn"],
            # "eng": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2024-02-13", "2025-02-13"),
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""@inproceedings{yu2025injongo,
  address = {Vienna, Austria},
  author = {Yu, Hao and Alabi, Jesujoba Oluwadara and Bukula, Andiswa and Zhuang, Jian Yun and Lee, En-Shiun Annie and Guge, Tadesse Kebede and Azime, Israel Abebe and Buzaaba, Happy and Sibanda, Blessing Kudzaishe and Kalipe, Godson Koffi and Mukiibi, Jonathan and Kabongo Kabenamualu, Salomon and Setaka, Mmasibidi and Ndolela, Lolwethu and Odu, Nkiruka and Mabuya, Rooweither and Muhammad, Shamsuddeen Hassan and Osei, Salomey and Samb, Sokhar and Klakow, Dietrich and Adelani, David Ifeoluwa},
  booktitle = {Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2025.acl-long.464},
  editor = {Che, Wanxiang and Nabende, Joyce and Shutova, Ekaterina and Pilehvar, Mohammad Taher},
  isbn = {979-8-89176-251-0},
  month = {jul},
  pages = {9429--9452},
  publisher = {Association for Computational Linguistics},
  title = {{INJONGO}: A Multicultural Intent Detection and Slot-filling Dataset for 16 {A}frican Languages},
  url = {https://aclanthology.org/2025.acl-long.464/},
  year = {2025},
}
""",
    )

    def dataset_transform(self, **kwargs: Any):
        """Convert HuggingFace splits to the lists of dicts expected by AbsTaskClassification."""
        transformed = {}

        for lang, splits in self.dataset.items():
            transformed[lang] = {}

            for split_name, ds in splits.items():
                # heuristic: locate the text and intent columns once per split
                cols = ds.column_names
                text_col = next(
                    (c for c in ["text", "utterance", "sentence"] if c in cols), None
                )
                label_col = next(
                    (c for c in ["label", "intent", "labels"] if c in cols), None
                )

                if text_col is None or label_col is None:
                    raise ValueError(
                        f"Couldn’t find text/label columns in InjongoIntent – {lang}-{split_name}"
                    )
                transformed[lang][split_name] = ds.rename_columns(
                    {text_col: "text", label_col: "label"}
                )

        self.dataset = transformed
