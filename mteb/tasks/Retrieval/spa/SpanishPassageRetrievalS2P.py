from __future__ import annotations

from mteb.abstasks.task_metadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
  abstract = {This paper describes a new test collection for passage retrieval from health-related Web resources in Spanish. The test collection contains 10,037 health-related documents in Spanish, 37 topics representing complex information needs formulated in a total of 167 natural language questions, and manual relevance assessments of text passages, pooled from multiple systems. This test collection is the first to combine search in a language beyond English, passage retrieval, and health-related resources and topics targeting the general public.},
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
