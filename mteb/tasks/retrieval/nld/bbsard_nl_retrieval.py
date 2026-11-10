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
@inproceedings{lotfi_bilingual_2025,
  title = {Bilingual {BSARD}: Extending Statutory Article Retrieval to {D}utch},
  author = {Lotfi, Ehsan  and
      Banar, Nikolay  and
      Yuzbashyan, Nerses  and
      Daelemans, Walter},
  editor = {Gokhan, Tuba  and
      Wang, Kexin  and
      Gurevych, Iryna  and
      Briscoe, Ted},
  booktitle = {Proceedings of the 1st Regulatory {NLP} Workshop ({R}eg{NLP} 2025)},
  month = jan,
  year = {2025},
  address = {Abu Dhabi, UAE},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2025.regnlp-1.3/},
  pages = {10--21},
}
""",
        prompt={
            "query": "Gegeven een juridische vraag, haal documenten op die kunnen helpen bij het beantwoorden van de vraag"
        },
    )
