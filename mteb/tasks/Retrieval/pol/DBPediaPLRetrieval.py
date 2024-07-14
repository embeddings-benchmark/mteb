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
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{Hasibi:2017:DVT,
 author =    {Hasibi, Faegheh and Nikolaev, Fedor and Xiong, Chenyan and Balog, Krisztian and Bratsberg, Svein Erik and Kotov, Alexander and Callan, Jamie},
 title =     {DBpedia-Entity V2: A Test Collection for Entity Search},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series =    {SIGIR '17},
 year =      {2017},
 pages =     {1265--1268},
 doi =       {10.1145/3077136.3080751},
 publisher = {ACM}
}""",
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 311.7007956561823,
                    "average_query_length": 35.45,
                    "num_documents": 4635922,
                    "num_queries": 400,
                    "average_relevant_docs_per_query": 38.215,
                }
            },
        },
    )
