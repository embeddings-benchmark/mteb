from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MLQuestionsRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="MLQuestions",
        dataset={
            "path": "mteb/MLQuestions",
            "revision": "99f13956087b5c2e21beeec8022a5f0f86d9f577",
        },
        reference="https://github.com/McGill-NLP/MLQuestions",
        description=(
            "MLQuestions is a domain adaptation dataset for the machine learning domain"
            "It consists of ML questions along with passages from Wikipedia machine learning pages (https://en.wikipedia.org/wiki/Category:Machine_learning)"
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=(
            "2021-01-01",
            "2021-03-31",
        ),  # The period here is for both wiki articles and queries
        domains=["Encyclopaedic", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{kulshreshtha-etal-2021-back,
  address = {Online and Punta Cana, Dominican Republic},
  author = {Kulshreshtha, Devang  and
Belfer, Robert  and
Serban, Iulian Vlad  and
Reddy, Siva},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  month = nov,
  pages = {7064--7078},
  publisher = {Association for Computational Linguistics},
  title = {Back-Training excels Self-Training at Unsupervised Domain Adaptation of Question Generation and Passage Retrieval},
  url = {https://aclanthology.org/2021.emnlp-main.566},
  year = {2021},
}
""",
    )
