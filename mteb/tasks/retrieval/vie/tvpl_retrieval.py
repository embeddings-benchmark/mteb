from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

TEST_SAMPLES = 2048


class TVPLRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TVPLRetrieval",
        description="A Vietnamese dataset for evaluating legal text retrieval. From Thu vien phap luat (TVPL) dataset: Optimizing Answer Generator in Vietnamese Legal Question Answering Systems Using Language Models.",
        reference="https://aclanthology.org/2020.coling-main.233.pdf",
        dataset={
            "path": "GreenNode/TVPL-Retrieval-VN",
            "revision": "6661dba4dfedff606537732d9f35f2c3738b081a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-07-29", "2025-07-30"),
        license="cc-by-sa-4.0",
        dialect=[],
        annotations_creators="human-annotated",
        domains=["Legal"],
        task_subtypes=["Question answering"],
        sample_creation="found",
        bibtex_citation=r"""
@article{10.1145/3732938,
  address = {New York, NY, USA},
  author = {Le, Huong and Luu, Ngoc and Nguyen, Thanh and Dao, Tuan and Dinh, Sang},
  doi = {10.1145/3732938},
  issn = {2375-4699},
  journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
  publisher = {Association for Computing Machinery},
  title = {Optimizing Answer Generator in Vietnamese Legal Question Answering Systems Using Language Models},
  url = {https://doi.org/10.1145/3732938},
  year = {2025},
}
""",
    )
