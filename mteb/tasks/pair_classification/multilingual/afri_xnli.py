from __future__ import annotations

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AfriXNLI(AbsTaskPairClassification):
    """Custom task for the AfriXNLI dataset."""

    metadata = TaskMetadata(
        name="AfriXNLI",
        description=(
            "Cross-lingual natural language inference dataset focusing on "
            "African languages."
        ),
        reference="https://github.com/masakhane-io/afri-xnli",
        dataset={
            "path": "mteb/AfriXNLI",  # dataset on the HF hub
            "revision": "92cff70b5f9bd7991176a47ba4203344fa710cfd",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "eng": ["eng-Latn"],
            "fra": ["fra-Latn"],
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
        },
        main_score="max_ap",
        date=("2020-01-01", "2020-12-31"),
        domains=["News", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
    )

    def dataset_transform(self, **kwargs):
        # The current HF dataset already has the correct schema:
        # sentence1, sentence2, labels with binary 0/1 labels.
        # Do not filter or remap labels.
        return
