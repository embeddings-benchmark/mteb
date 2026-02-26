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
@article{zhu2024longembed,
  author = {Zhu, Dawei and Wang, Liang and Yang, Nan and Song, Yifan and Wu, Wenhao and Wei, Furu and Li, Sujian},
  journal = {arXiv preprint arXiv:2404.12096},
  title = {LongEmbed: Extending Embedding Models for Long Context Retrieval},
  year = {2024},
}
""",
    )
