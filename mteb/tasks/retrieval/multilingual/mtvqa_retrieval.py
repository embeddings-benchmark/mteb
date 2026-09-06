from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX = r"""
@article{tang2024mtvqa,
  author = {Tang, Jingqun and Liu, Qi and Ye, Yongjie and Lu, Jinghui and Wei, Shu and Lin, Chunhui and Li, Wanqing and Mahmood, Mohamad Fitri Faiz Bin and Feng, Hao and Zhao, Zhen and Wang, Yanjie and Liu, Yuliang and Liu, Hao and Bai, Xiang and Huang, Can},
  journal = {arXiv preprint arXiv:2405.11985},
  title = {{MTVQA}: Benchmarking Multilingual Text-Centric Visual Question Answering},
  year = {2024},
}
"""


class MTVQAIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MTVQAIT2TRetrieval",
        description=(
            "Retrieve the answer to a question about the text inside an image, across "
            "nine languages. The query is the image together with the question, and the "
            "corpus is the answers for that language. mteb has thirteen it2t tasks and "
            "none of them is multilingual, so this scores whether a model reads text in "
            "a picture in a language other than English. Built from the official test "
            "split. Answers repeat across questions, because different images can share "
            "a short answer such as a price or a name, so the corpus is deduplicated by "
            "answer text and every question with that answer points at the surviving "
            "document rather than dropping the question. Construction script: "
            "scripts/data/mtvqa_retrieval/create_data.py."
        ),
        reference="https://arxiv.org/abs/2405.11985",
        dataset={
            "path": "vnahata/MTVQA-it2t-retrieval",
            "revision": "f2d2ab47cf576e94ef89807d9c2dd8012515cf84",
        },
        type="Any2AnyMultilingualRetrieval",
        category="it2t",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs={
            "ara": ["ara-Arab"],
            "deu": ["deu-Latn"],
            "fra": ["fra-Latn"],
            "ita": ["ita-Latn"],
            "jpn": ["jpn-Jpan"],
            "kor": ["kor-Hang"],
            "rus": ["rus-Cyrl"],
            "tha": ["tha-Thai"],
            "vie": ["vie-Latn"],
        },
        main_score="accuracy",
        date=("2023-01-01", "2024-05-20"),
        domains=["Scene", "Written"],
        task_subtypes=["Rendered Texts Understanding"],
        license="cc-by-nc-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        prompt={"query": "Answer the question about the text in this image."},
        bibtex_citation=_BIBTEX,
        is_beta=True,
    )
