import logging

from mteb.abstasks import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


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

    input1_column_name: str = "audio1"
    input2_column_name: str = "audio2"
    label_column_name: str = "label"
