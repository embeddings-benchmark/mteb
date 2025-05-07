from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class NQ(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ",
        dataset={
            "path": "mteb/nq",
            "revision": "b774495ed302d8c44a3a7ea25c90dbce03968f31",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{47761,
  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
and Slav Petrov},
  journal = {Transactions of the Association of Computational
Linguistics},
  title = {Natural Questions: a Benchmark for Question Answering Research},
  year = {2019},
}
""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
    )


class NQHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQHardNegatives",
        dataset={
            "path": "mteb/NQ_test_top_250_only_w_correct-v2",
            "revision": "d700fe4f167a5db8e6c9b03e8c26e7eaf66faf97",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{47761,
  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh
and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee
and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le
and Slav Petrov},
  journal = {Transactions of the Association of Computational
Linguistics},
  title = {Natural Questions: a Benchmark for Question Answering Research},
  year = {2019},
}
""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NQ"],
    )
