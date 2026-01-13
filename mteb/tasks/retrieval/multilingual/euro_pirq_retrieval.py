from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "fi": ["fin-Latn"],
    "pt": ["por-Latn"],
}


class EuroPIRQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EuroPIRQRetrieval",
        description="The EuroPIRQ retrieval dataset is a multilingual collection designed for evaluating retrieval and cross-lingual retrieval tasks. Dataset contains 10,000 parallel passages & 100 parallel queries (synthetic) in three languages: English, Portuguese, and Finnish, constructed from the European Union's DGT-Acquis corpus.",
        reference="https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval",
        dataset={
            "path": "eherra/EuroPIRQ-retrieval",
            "revision": "59225ed25fbcea2185e1acbc8c3c80f1a8cd8341",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2025-12-01", "2025-12-31"),
        domains=["Legal"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="LM-generated and reviewed",
        dialect=[],
        sample_creation="found",
        is_public=True,
        bibtex_citation=r"""
@misc{eherra_2025_europirq,
  author = { {Elias Herranen} },
  publisher = { Hugging Face },
  title = { EuroPIRQ: European Parallel Information Retrieval Queries },
  url = { https://huggingface.co/datasets/eherra/EuroPIRQ-retrieval },
  year = {2025},
}
""",
    )
