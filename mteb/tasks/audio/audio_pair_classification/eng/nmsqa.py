from __future__ import annotations

import logging

from datasets import DatasetDict, concatenate_datasets

from mteb.abstasks.audio.abs_task_audio_pair_classification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class NMSQAPairClassification(AbsTaskAudioPairClassification):
    metadata = TaskMetadata(
        name="NMSQAPairClassification",
        description="A textless Q&A dataset. Given a pair of audio question and audio answer, is the answer relevant to the question?",
        reference="https://www.researchgate.net/publication/311458869_FMA_A_Dataset_For_Music_Analysis",
        dataset={
            "path": "GSQA/NMSQA-test",
            "revision": "39b80eb7f5135c0571db23049a2ba4837b41b0cf",
        },
        type="AudioPairClassification",
        category="a2a",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2016-01-01", "2016-12-31"),
        domains=["Spoken"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{lin2022dualdiscretespokenunit,
  archiveprefix = {arXiv},
  author = {Guan-Ting Lin and Yung-Sung Chuang and Ho-Lam Chung and Shu-wen Yang and Hsuan-Jui Chen and Shuyan Dong and Shang-Wen Li and Abdelrahman Mohamed and Hung-yi Lee and Lin-shan Lee},
  eprint = {2203.04911},
  primaryclass = {cs.CL},
  title = {DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering},
  url = {https://arxiv.org/abs/2203.04911},
  year = {2022},
}
""",
    )

    # Override default column name in the subclass

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def _extract_waveform_from_df(
        self,
        df,
        audio1_name: str = "question_audio_path",
        audio2_name: str = "content_segment_audio_path",
    ):
        df.loc[:, audio1_name] = df.apply(lambda row: row[audio1_name]["array"], axis=1)
        df.loc[:, audio2_name] = df.apply(lambda row: row[audio2_name]["array"], axis=1)

    def dataset_transform(self):
        ds = self.dataset["test"]

        ds = ds.shuffle(self.seed)

        # split into similar and dissimilar halves
        half = len(ds) // 2
        ds_sim = ds.select(range(half))
        ds_dissim = ds.select(range(half, len(ds)))

        # add label to similar pairs
        ds_sim = ds_sim.add_column("label", [1] * len(ds_sim))

        # extract waveforms for similar pairs
        ds_sim = ds_sim.map(
            lambda row: {
                "question_audio_path": row["question_audio_path"]["array"],
                "content_segment_audio_path": row["content_segment_audio_path"][
                    "array"
                ],
            },
            batched=False,
        )

        ds_dissim = ds_dissim.map(
            lambda row, idx: {
                "question_audio_path": row["question_audio_path"]["array"],
                "content_segment_audio_path": ds_dissim[(idx + 1) % len(ds_dissim)][
                    "content_segment_audio_path"
                ]["array"],
                "label": 0,
            },
            with_indices=True,
        )

        ds_combined = concatenate_datasets([ds_sim, ds_dissim])

        ds_combined = ds_combined.shuffle(seed=self.seed)

        ds_combined = ds_combined.rename_columns(
            {"question_audio_path": "audio1", "content_segment_audio_path": "audio2"}
        )

        # wrap in DatasetDict
        self.dataset = DatasetDict({"test": ds_combined})
