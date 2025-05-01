from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPediaPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-PL",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "clarin-knext/dbpedia-pl",
            "revision": "76afe41d9af165cc40999fcaa92312b8b012064a",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-01-01"),  # best guess: based on publication date
        domains=["Written", "Encyclopaedic"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{Hasibi:2017:DVT,
  author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
  booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  doi = {10.1145/3077136.3080751},
  pages = {1265--1268},
  publisher = {ACM},
  series = {SIGIR '17},
  title = {DBpedia-Entity V2: A Test Collection for Entity Search},
  year = {2017},
}
""",
        adapted_from=["DBPedia"],
    )


class DBPediaPLHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia-PLHardNegatives",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "mteb/DBPedia_PL_test_top_250_only_w_correct-v2",
            "revision": "bebc2b5c8f73cd6ba9d2a4664d5f3769e6ad557a",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-01-01"),  # best guess: based on publication date
        domains=["Written", "Encyclopaedic"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{Hasibi:2017:DVT,
  author = {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
  booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  doi = {10.1145/3077136.3080751},
  pages = {1265--1268},
  publisher = {ACM},
  series = {SIGIR '17},
  title = {DBpedia-Entity V2: A Test Collection for Entity Search},
  year = {2017},
}
""",
        adapted_from=["DBPedia"],
    )
