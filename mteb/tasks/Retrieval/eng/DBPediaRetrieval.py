from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "mteb/dbpedia",
            "revision": "c0f706b76e590d620bd6618b3ca8efdd34e2d659",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev", "test"],
        eval_langs=["eng-Latn"],
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
        stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 1122.7690155333814,
                    "average_query_length": 48.7264325323475,
                    "num_documents": 48605,
                    "num_queries": 541,
                    "average_relevant_docs_per_query": 1.3752310536044363,
                }
            },
        },
    )
