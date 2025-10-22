from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class SpanishPassageRetrievalS2P(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SpanishPassageRetrievalS2P",
        description="Test collection for passage retrieval from health-related Web resources in Spanish.",
        reference="https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/",
        dataset={
            "path": "mteb/SpanishPassageRetrievalS2P",
            "revision": "35c4548ee4e971ea81bcd3f4b431db6ff599ff88",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{10.1007/978-3-030-15719-7_19,
  address = {Cham},
  author = {Kamateri, Eleni
and Tsikrika, Theodora
and Symeonidis, Spyridon
and Vrochidis, Stefanos
and Minker, Wolfgang
and Kompatsiaris, Yiannis},
  booktitle = {Advances in Information Retrieval},
  editor = {Azzopardi, Leif
and Stein, Benno
and Fuhr, Norbert
and Mayr, Philipp
and Hauff, Claudia
and Hiemstra, Djoerd},
  isbn = {978-3-030-15719-7},
  pages = {148--154},
  publisher = {Springer International Publishing},
  title = {A Test Collection for Passage Retrieval Evaluation of Spanish Health-Related Resources},
  year = {2019},
}
""",
    )
