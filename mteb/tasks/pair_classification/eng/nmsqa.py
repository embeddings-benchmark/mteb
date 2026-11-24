from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class NMSQAPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="NMSQAPairClassification",
        description="A textless Q&A dataset. Given a pair of audio question and audio answer, is the answer relevant to the question?",
        reference="https://www.researchgate.net/publication/311458869_FMA_A_Dataset_For_Music_Analysis",
        dataset={
            "path": "mteb/NMSQAPairClassification",
            "revision": "d33eea302d8a64c0a2ee094cf39d77c0814fe87a",
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
@inproceedings{lin2022dualdiscretespokenunit,
  author = {Lin, Guan-Ting and Chuang, Yung-Sung and Chung, Ho-Lam and Yang, Shu-wen and Chen, Hsuan-Jui and Dong, Shuyan Annie and Li, Shang-Wen and Mohamed, Abdelrahman and Lee, Hung-yi and Lee, Lin-shan},
  booktitle = {Interspeech 2022},
  collection = {interspeech_2022},
  doi = {10.21437/interspeech.2022-612},
  month = {Sept},
  pages = {5165–5169},
  publisher = {ISCA},
  series = {interspeech_2022},
  title = {DUAL: Discrete Spoken Unit Adaptive Learning for Textless Spoken Question Answering},
  url = {https://doi.org/10.21437/interspeech.2022-612},
  year = {2022},
}
""",
    )

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"
