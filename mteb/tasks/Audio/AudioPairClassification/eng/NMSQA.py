from __future__ import annotations

import random

import datasets
import pandas as pd

from mteb.abstasks.Audio.AbsTaskAudioPairClassification import (
    AbsTaskAudioPairClassification,
)
from mteb.abstasks.TaskMetadata import TaskMetadata

random.seed(42)


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
        category="t2t",  # no audio category yet
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        domains=["Spoken"],
        task_subtypes=["Question answering"],
        license="not specified",
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation="""@misc{lin2022dualdiscretespokenunit,
      title={DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering},
      author={Guan-Ting Lin and Yung-Sung Chuang and Ho-Lam Chung and Shu-wen Yang and Hsuan-Jui Chen and Shuyan Dong and Shang-Wen Li and Abdelrahman Mohamed and Hung-yi Lee and Lin-shan Lee},
      year={2022},
      eprint={2203.04911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.04911},
}
        """,
        descriptive_stats={"n_samples": {"test": 171}},
    )

    # Override default column name in the subclass

    audio1_column_name: str = "audio1"
    audio2_column_name: str = "audio2"
    label_column_name: str = "label"
    samples_per_label: int = 2

    def _extract_waveform_from_df(
        self,
        df: pd.DataFrame,
        audio1_name: str = "question_audio_path",
        audio2_name: str = "content_segment_audio_path",
    ):
        df.loc[:, audio1_name] = df.apply(lambda row: row[audio1_name]["array"], axis=1)
        df.loc[:, audio2_name] = df.apply(lambda row: row[audio2_name]["array"], axis=1)

    def dataset_transform(self):
        df = pd.DataFrame(self.dataset["test"])

        # shuffle and split dataset by row
        df = df.sample(frac=1)
        df_sim = df.iloc[: len(df) // 2, :]
        df_dissim = df.iloc[len(df) // 2 :, :]

        similar_pairs = []
        dissimilar_pairs = []

        # match question and answer pairs for similar
        columns_to_keep = ["question_audio_path", "content_segment_audio_path"]
        df_sim = df_sim[columns_to_keep]
        df_sim["label"] = 1
        self._extract_waveform_from_df(df_sim)
        self._extract_waveform_from_df(df_dissim)

        similar_pairs = df_sim.values.tolist()

        num_similar = len(similar_pairs)
        print("Similar pairs: ", num_similar)

        # shuffle and mismatch question and answer pairs for dissimilar
        df_dissim.sample(frac=1)
        dissimilar_pairs = [
            [
                df_dissim.iloc[i]["question_audio_path"],
                df_dissim.iloc[(i + 1) % len(df_dissim)]["content_segment_audio_path"],
                0,
            ]
            for i in range(len(df_dissim))
        ]

        pairs = similar_pairs + dissimilar_pairs
        print("Number of pairs: ", len(pairs))
        random.shuffle(pairs)

        audio1, audio2, label = zip(*pairs[:10])
        audio1 = list(audio1)
        audio2 = list(audio2)
        label = list(label)

        print("Generating dataset: ")
        HF_ds = datasets.Dataset.from_dict(
            {"audio1": audio1, "audio2": audio2, "label": label}
        )

        # convert back to HF dataset
        self.dataset = datasets.DatasetDict({"test": HF_ds})
        print("done!")
