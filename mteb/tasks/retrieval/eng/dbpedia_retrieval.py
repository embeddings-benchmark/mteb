from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_dbpedia_metadata = dict(
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="ndcg_at_10",
    date=("2017-01-01", "2017-01-01"),  # best guess: based on publication date
    domains=["Written", "Encyclopaedic"],
    task_subtypes=[],
    license="mit",
    annotations_creators="derived",
    dialect=[],
    sample_creation="found",
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
)


class DBPedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPedia",
        description="DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base",
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "mteb/dbpedia",
            "revision": "c0f706b76e590d620bd6618b3ca8efdd34e2d659",
        },
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
        **_dbpedia_metadata,
    )


class DBPediaHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPediaHardNegatives",
        description=(
            "DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
        ),
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "mteb/DBPedia_test_top_250_only_w_correct-v2",
            "revision": "943ec7fdfef3728b2ad1966c5b6479ff9ffd26c9",
        },
        superseded_by="DBPediaHardNegatives.v2",
        adapted_from=["DBPedia"],
        **_dbpedia_metadata,
    )


class DBPediaHardNegativesV2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DBPediaHardNegatives.v2",
        description=(
            "DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. "
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. You can get more information on the effect of different prompt in the [PR](https://github.com/embeddings-benchmark/mteb/pull/3469#issuecomment-3436467106)"
        ),
        reference="https://github.com/iai-group/DBpedia-Entity/",
        dataset={
            "path": "mteb/DBPedia_test_top_250_only_w_correct-v2",
            "revision": "943ec7fdfef3728b2ad1966c5b6479ff9ffd26c9",
        },
        adapted_from=["DBPedia"],
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
        **_dbpedia_metadata,
    )
