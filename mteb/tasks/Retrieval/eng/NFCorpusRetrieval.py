from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NFCorpus(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NFCorpus",
        dataset={
            "path": "mteb/nfcorpus",
            "revision": "ec0fa4fe99da2ff19ca1214b7966684033a58814",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Academic", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""@inproceedings{boteva2016,
  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  journal = {Proceedings of the 38th European Conference on Information Retrieval},
  journal-abbrev = {ECIR},
  year = {2016},
  city = {Padova},
  country = {Italy},
  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf}
}""",
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
    )
