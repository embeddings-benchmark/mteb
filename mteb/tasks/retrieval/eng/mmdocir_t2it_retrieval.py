from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

DOMAINS = [
    "academic_paper",
    "admin_industry",
    "brochure",
    "financial_report",
    "government",
    "guidebook",
    "laws",
    "news",
    "research_report",
    "tutorial_workshop",
]


class MMDocIRT2ITRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMDocIRT2ITRetrieval",
        description="MMDocIR evaluation set includes 313 long documents averaging 65.1 pages, categorized into ten main domains: research reports, administration&industry, tutorials&workshops, academic papers, brochures, financial reports, guidebooks, government documents, laws, and news articles. Different domains feature distinct distributions of multi-modal information. Overall, the modality distribution is: Text (60.4%), Image (18.8%), Table (16.7%), and other modalities (4.1%).",
        reference="https://doi.org/10.18653/v1/2025.emnlp-main.1576",
        dataset={
            "path": "mteb/MMDocIRT2ITRetrieval",
            "revision": "ef97f91ffcd80abf10dda2f72f5971c18c5d5e74",
        },
        type="DocumentUnderstanding",
        category="t2it",
        eval_splits=["train"],
        eval_langs={domain: ["eng-Latn"] for domain in DOMAINS},
        main_score="recall_at_5",
        date=("2025-01-01", "2025-01-01"),
        domains=["Academic", "Non-fiction", "Government", "Legal", "News"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{dong2025mmdocirbenchmarkingmultimodalretrieval,
  author = {Dong, Kuicai and Chang, Yujing and Goh Xin Deik, Derrick and Li, Dexun and Tang, Ruiming and Liu, Yong},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2025.emnlp-main.1576},
  pages = {30959–30993},
  publisher = {Association for Computational Linguistics},
  title = {MMDocIR: Benchmarking Multimodal Retrieval for Long Documents},
  url = {https://doi.org/10.18653/v1/2025.emnlp-main.1576},
  year = {2025},
}
""",
    )
