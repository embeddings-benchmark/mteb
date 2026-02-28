from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class InjongoIntent(MultilingualTask, AbsTaskClassification):
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
        reference="https://arxiv.org/abs/2502.09814",
        dataset={
            "path": "masakhane/InjongoIntent",
            # exact commit that matches the paper; bump when the HF repo updates
            "revision": "fe4be3882a1614161dfe231ec793197bb74f4b44",
        },
        type="Classification",
        category="s2s",
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
            "orm": ["orm-Ethi"],
            "sna": ["sna-Latn"],
            "sot": ["sot-Latn"],
            "swa": ["swa-Latn"],
            "twi": ["twi-Latn"],
            "wol": ["wol-Latn"],
            "xho": ["xho-Latn"],
            "yor": ["yor-Latn"],
            "zul": ["zul-Latn"],
            "eng": ["eng-Latn"],
        },
        main_score="accuracy",
        date=("2024-02-13", "2025-02-13"),
        form=["spoken"],
        domains=["Spoken"],
        task_subtypes=["Intent classification"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        socioeconomic_status="mixed",
        dialect=[],
        text_creation="created",
        bibtex_citation="""@article{yu2025injongo,
  title     = {INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages},
  author    = {Yu, Hao and Alabi, Jesujoba O. and Bukula, Andiswa and et al.},
  journal   = {arXiv preprint arXiv:2502.09814},
  year      = {2025}
}""",
    )


    def dataset_transform(self):
        """Convert HuggingFace splits to the lists of dicts expected by AbsTaskClassification."""
        transformed = {}

        for lang, splits in self.dataset.items():
            transformed[lang] = {}

            for split_name, ds in splits.items():
                # heuristic: locate the text and intent columns once per split
                cols = ds.column_names
                text_col = next((c for c in ["text", "utterance", "sentence"] if c in cols), None)
                label_col = next((c for c in ["label", "intent", "labels"] if c in cols), None)

                if text_col is None or label_col is None:
                    raise ValueError(
                        f"Couldn’t find text/label columns in InjongoIntent – {lang}-{split_name}"
                    )

                # transformed[lang][split_name] = [
                #     {"text": txt, "label": lbl}
                #     for txt, lbl in zip(ds[text_col], ds[label_col])
                # ]
                ds = ds.rename_columns({text_col: "text", label_col: "label"})
                transformed[lang][split_name] = ds

        self.dataset = transformed
