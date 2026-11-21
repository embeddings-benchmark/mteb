from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class VieQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VieQuADRetrieval",
        description="A Vietnamese dataset for evaluating Machine Reading Comprehension from Wikipedia articles.",
        reference="https://aclanthology.org/2020.coling-main.233.pdf",
        dataset={
            "path": "mteb/VieQuADRetrieval",
            "revision": "1700b6dc1c2d9415fcb6f0350b9493db825e2337",
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
        bibtex_citation=r"""
@inproceedings{nguyen-etal-2020-vietnamese,
  address = {Barcelona, Spain (Online)},
  author = {Nguyen, Kiet  and
Nguyen, Vu  and
Nguyen, Anh  and
Nguyen, Ngan},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  doi = {10.18653/v1/2020.coling-main.233},
  editor = {Scott, Donia  and
Bel, Nuria  and
Zong, Chengqing},
  month = dec,
  pages = {2595--2605},
  publisher = {International Committee on Computational Linguistics},
  title = {A Vietnamese Dataset for Evaluating Machine Reading Comprehension},
  url = {https://aclanthology.org/2020.coling-main.233},
  year = {2020},
}
""",
    )
