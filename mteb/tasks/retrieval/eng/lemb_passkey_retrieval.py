from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBPasskeyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LEMBPasskeyRetrieval",
        dataset={
            "path": "mteb/LEMBPasskeyRetrieval",
            "revision": "86ac5a6437198a94ecd666440c78b3e78f258274",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description="passkey subset of dwzhu/LongEmbed dataset.",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[
            "test_256",
            "test_512",
            "test_1024",
            "test_2048",
            "test_4096",
            "test_8192",
            "test_16384",
            "test_32768",
        ],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_1",
        date=("2000-01-01", "2023-12-31"),
        domains=["Fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{zhu2024longembed,
  address = {Miami, Florida, USA},
  author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2024.emnlp-main.47},
  editor = {Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung},
  month = {nov},
  pages = {802--816},
  publisher = {Association for Computational Linguistics},
  title = {{L}ong{E}mbed: Extending Embedding Models for Long Context Retrieval},
  url = {https://aclanthology.org/2024.emnlp-main.47/},
  year = {2024},
}
""",
    )
