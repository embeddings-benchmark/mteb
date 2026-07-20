from typing import Any

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

# Corrupt/truncated WAV in the HF snapshot (no RIFF data chunk).
_CORRUPTED_AUDIO_FILENAME = "s05 (3).wav"


class AESDD(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AESDD",
        description="Acted Emotional Speech Dynamic Database (AESDD): emotion classification of acted Greek speech into one of 5 classes: anger, disgust, fear, happiness, and sadness.",
        reference="https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/",
        dataset={
            "path": "mteb/AESDD",
            "revision": "2ab5cc0f7126088d11d1752747cdd1d7f74625c6",
        },
        type="AudioClassification",
        category="a2c",
        eval_splits=["train"],
        eval_langs=["ell-Grek"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["Speech"],
        task_subtypes=["Emotion classification"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="created",
        bibtex_citation=r"""
@article{vryzas2018speech,
  author = {Vryzas, Nikolaos and Kotsakis, Rigas and Liatsou, Aikaterini and Dimoulas, Charalampos A and Kalliris, George},
  journal = {Journal of the Audio Engineering Society},
  number = {6},
  pages = {457--467},
  publisher = {Audio Engineering Society},
  title = {Speech emotion recognition for performance interaction},
  volume = {66},
  year = {2018},
}
""",
    )

    input_column_name: str = "audio"
    label_column_name: str = "label"

    is_cross_validation: bool = True

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Drop the one corrupt WAV that cannot be decoded."""
        from datasets import Audio

        for split in self.dataset:
            split_ds = self.dataset[split].cast_column(
                self.input_column_name, Audio(decode=False)
            )
            split_ds = split_ds.filter(
                lambda example, bad=_CORRUPTED_AUDIO_FILENAME: not example[
                    self.input_column_name
                ]["path"].endswith(bad),
                num_proc=num_proc,
            )
            self.dataset[split] = split_ds.cast_column(
                self.input_column_name, Audio(decode=True)
            )
