from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BBSARDNLRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="bBSARDNLRetrieval",
        description="Building on the Belgian Statutory Article Retrieval Dataset (BSARD) in French, we introduce the "
        "bilingual version of this dataset, bBSARD. The dataset contains parallel Belgian statutory "
        "articles in both French and Dutch, along with legal questions from BSARD and their Dutch "
        "translation.",
        reference="https://aclanthology.org/2025.regnlp-1.3.pdf",
        dataset={
            "path": "clips/mteb-nl-bbsard",
            "revision": "52027c212ba9765a3e9737c9cbf9a06ae83cbb93",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{lotfi2025bilingual,
  author = {Lotfi, Ehsan and Banar, Nikolay and Yuzbashyan, Nerses and Daelemans, Walter},
  journal = {COLING 2025},
  pages = {10},
  title = {Bilingual BSARD: Extending Statutory Article Retrieval to Dutch},
  year = {2025},
}
""",
        prompt={
            "query": "Gegeven een juridische vraag, haal documenten op die kunnen helpen bij het beantwoorden van de vraag"
        },
    )
