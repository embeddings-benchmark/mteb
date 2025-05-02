from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class VieQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VieQuADRetrieval",
        description="A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.",
        reference="https://aclanthology.org/2020.coling-main.233.pdf",
        dataset={
            "path": "mteb/VieQuADRetrieval",
            "revision": "f956535e394d5f2b4334de447151e3b237ef19d1",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2022-03-02", "2022-03-02"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{nguyen-etal-2020-vietnamese,
title = "A Vietnamese Dataset for Evaluating Machine Reading Comprehension",
author = "Nguyen, Kiet  and
    Nguyen, Vu  and
    Nguyen, Anh  and
    Nguyen, Ngan",
editor = "Scott, Donia  and
    Bel, Nuria  and
    Zong, Chengqing",
booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
month = dec,
year = "2020",
address = "Barcelona, Spain (Online)",
publisher = "International Committee on Computational Linguistics",
url = "https://aclanthology.org/2020.coling-main.233",
doi = "10.18653/v1/2020.coling-main.233",
pages = "2595--2605"}""",
    )
