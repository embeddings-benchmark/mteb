from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class ZacLegalTextRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ZacLegalTextRetrieval",
        description="Zalo Legal Text documents",
        reference="https://challenge.zalo.ai",
        dataset={
            "path": "GreenNode/zalo-ai-legal-text-retrieval-vn",
            "revision": "910766554633e8da014e88f54988705dde7ecaac",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["vie-Latn"],
        main_score="ndcg_at_10",
        date=("2025-03-16", "2025-03-16"),
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{10.1007/978-981-95-1746-6_17,
  abstract = {Information retrieval often comes in plain text, lacking semi-structured text such as HTML and markdown, retrieving data that contains rich format such as table became non-trivial. In this paper, we tackle this challenge by introducing a new dataset, GreenNode Table Retrieval VN (GN-TRVN), which is collected from a massive corpus, a wide range of topics, and a longer context compared to ViQuAD2.0. To evaluate the effectiveness of our proposed dataset, we introduce two versions, M3-GN-VN and M3-GN-VN-Mixed, by fine-tuning the M3-Embedding model on this dataset. Experimental results show that our models consistently outperform the baselines, including the base model, across most evaluation criteria on various datasets such as VieQuADRetrieval, ZacLegalTextRetrieval, and GN-TRVN. In general, we release a more comprehensive dataset and two model versions that improve response performance for Vietnamese Markdown Table Retrieval.},
  address = {Singapore},
  author = {Pham, Bao Loc
and Hoang, Quoc Viet
and Luu, Quy Tung
and Vo, Trong Thu},
  booktitle = {Proceedings of the Fifth International Conference on Intelligent Systems and Networks},
  isbn = {978-981-95-1746-6},
  pages = {153--163},
  publisher = {Springer Nature Singapore},
  title = {GN-TRVN: A Benchmark for Vietnamese Table Markdown Retrieval Task},
  year = {2026},
}
""",
    )
