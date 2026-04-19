from __future__ import annotations

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

from .human_animal_cartoon_pc import _build_pair_dataset, _generate_pairs


class RAVDESSAVPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="RAVDESSAVPairClassification",
        description=(
            "Pair classification on the RAVDESS dataset: "
            "determining whether two video clips express the same emotion "
            "from 8 categories (neutral, calm, happy, sad, angry, fearful, "
            "surprise, disgust) in acted speech and song performances."
        ),
        reference="https://doi.org/10.1371/journal.pone.0196391",
        dataset={
            "path": "mteb/RAVDESS_AV",
            "revision": "13af08387c3ce5e86c179a3718eb158669268d65",
        },
        type="VideoPairClassification",
        category="v2v",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-05-16", "2018-05-16"),
        domains=["Spoken"],
        task_subtypes=["Emotion classification"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=["eng-US", "eng-CA"],
        modalities=["video"],
        sample_creation="created",
        bibtex_citation=r"""
@article{10.1371/journal.pone.0196391,
  author = {Livingstone, Steven R. AND Russo, Frank A.},
  doi = {10.1371/journal.pone.0196391},
  journal = {PLOS ONE},
  month = {05},
  number = {5},
  pages = {1-35},
  publisher = {Public Library of Science},
  title = {The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English},
  url = {https://doi.org/10.1371/journal.pone.0196391},
  volume = {13},
  year = {2018},
}
""",
        contributed_by="stef41",
        is_beta=True,
    )

    input1_column_name: str = "video1"
    input2_column_name: str = "video2"
    label_column_name: str = "label"

    def dataset_transform(self, num_proc: int | None = None, **kwargs) -> None:
        import random

        rng = random.Random(42)
        for split in self.metadata.eval_splits:
            ds = self.dataset[split]
            pairs = _generate_pairs(ds["emotion"], rng)
            self.dataset[split] = _build_pair_dataset(ds, pairs)
