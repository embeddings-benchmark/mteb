from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GermanDPR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GermanDPR",
        description="GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer",
        reference="https://huggingface.co/datasets/deepset/germandpr",
        dataset={
            "path": "mteb/GermanDPR",
            "revision": "64a4860e55ba6d8fcb923d5306d08e08b1c72794",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=("2020-05-19", "2021-04-26"),
        domains=["Written", "Non-fiction", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{möller2021germanquad,
  address = {Punta Cana, Dominican Republic},
  author = {M{\"o}ller, Timo and Risch, Julian and Pietsch, Malte},
  booktitle = {Proceedings of the 3rd Workshop on Machine Reading for Question Answering},
  doi = {10.18653/v1/2021.mrqa-1.4},
  editor = {Fisch, Adam and Talmor, Alon and Chen, Danqi and Choi, Eunsol and Seo, Minjoon and Lewis, Patrick and Jia, Robin and Min, Sewon},
  month = {nov},
  pages = {42--50},
  publisher = {Association for Computational Linguistics},
  title = {{G}erman{Q}u{AD} and {G}erman{DPR}: Improving Non-{E}nglish Question Answering and Passage Retrieval},
  url = {https://aclanthology.org/2021.mrqa-1.4/},
  year = {2021},
}
""",
    )
